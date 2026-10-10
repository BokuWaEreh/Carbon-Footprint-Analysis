"""
Carbon Footprint ML Insights
----------------------------
Reads your CarbonData export (CSV) and writes three CSV files that you can
import into Power BI:

  1. forecast_monthly.csv  - monthly actual CO2 + 6-month forecast with confidence band
  2. anomalies.csv         - every record with an anomaly flag and score (Isolation Forest)
  3. activity_clusters.csv - activities grouped into Low / Medium / High impact (K-Means)

Usage:
    pip install pandas scikit-learn statsmodels
    python ml_insights.py carbon_data.csv            # outputs go to ./ml_output
    python ml_insights.py carbon_data.csv my_folder  # custom output folder

Expected columns (names are matched case-insensitively):
    a date column (any column whose name contains "date"), Location, Category,
    Activity, Quantity, CO2_Emission_kg
"""
import sys, os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HORIZON = 6  # months to forecast


# ------------------------------------------------------------------ load
def load(path):
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}

    def find(*names, contains=None):
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        if contains:
            for k, v in lower.items():
                if contains in k:
                    return v
        return None

    cols = {
        "date": find("date", contains="date"),
        "location": find("location"),
        "category": find("category"),
        "activity": find("activity"),
        "quantity": find("quantity"),
        "co2": find("co2_emission_kg", contains="co2"),
    }
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        sys.exit(f"Could not find column(s) for: {missing}\nFound columns: {list(df.columns)}")

    out = pd.DataFrame({k: df[v] for k, v in cols.items()})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["co2"] = pd.to_numeric(out["co2"], errors="coerce")
    before = len(out)
    out = out.dropna(subset=["date", "co2"]).reset_index(drop=True)
    if len(out) < before:
        print(f"Dropped {before - len(out)} rows with missing date/CO2")
    return out


# ------------------------------------------------------------------ 1. forecast
def forecast(df):
    m = (df.set_index("date")["co2"].resample("MS").sum())
    n = len(m)
    print(f"Monthly history: {n} months")
    if n < 4:
        sys.exit("Need at least 4 months of data to forecast.")

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    seasonal = "add" if n >= 24 else None
    model = ExponentialSmoothing(m, trend="add", damped_trend=True,
                                 seasonal=seasonal,
                                 seasonal_periods=12 if seasonal else None,
                                 initialization_method="estimated").fit()
    fc = model.forecast(HORIZON)
    resid_sd = float(np.std(m - model.fittedvalues, ddof=1))
    # interval widens with horizon
    h = np.arange(1, HORIZON + 1)
    width = 1.96 * resid_sd * np.sqrt(h)
    fc_df = pd.DataFrame({"Month": fc.index, "Actual": np.nan,
                          "Forecast": fc.values,
                          "Lower": np.maximum(fc.values - width, 0),
                          "Upper": fc.values + width, "Type": "Forecast"})
    hist = pd.DataFrame({"Month": m.index, "Actual": m.values, "Forecast": np.nan,
                         "Lower": np.nan, "Upper": np.nan, "Type": "Actual"})
    # connect the lines: last actual point also starts the forecast series
    hist.loc[hist.index[-1], ["Forecast", "Lower", "Upper"]] = [m.iloc[-1]] * 3
    out = pd.concat([hist, fc_df], ignore_index=True)
    mape = float(np.mean(np.abs((m - model.fittedvalues) / m.replace(0, np.nan))) * 100)
    print(f"Forecast model: Holt-Winters ({'seasonal' if seasonal else 'trend only'}), "
          f"in-sample MAPE {mape:.1f}%")
    return out


# ------------------------------------------------------------------ 2. anomalies
def anomalies(df):
    d = df.copy()
    grp = d.groupby("activity")["co2"]
    d["co2_z_in_activity"] = (d["co2"] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    d["co2_z_in_activity"] = d["co2_z_in_activity"].fillna(0)
    feats = d[["quantity", "co2", "co2_z_in_activity"]].fillna(0)
    X = StandardScaler().fit_transform(feats)
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42).fit(X)
    d["AnomalyScore"] = -iso.score_samples(X)          # higher = more unusual
    d["IsAnomaly"] = np.where(iso.predict(X) == -1, "Anomaly", "Normal")
    d["RecordID"] = np.arange(1, len(d) + 1)
    out = d.rename(columns={"date": "Date", "location": "Location", "category": "Category",
                            "activity": "Activity", "quantity": "Quantity", "co2": "CO2_Emission_kg"})
    out = out[["RecordID", "Date", "Location", "Category", "Activity", "Quantity",
               "CO2_Emission_kg", "AnomalyScore", "IsAnomaly"]]
    print(f"Anomalies flagged: {(out.IsAnomaly == 'Anomaly').sum()} of {len(out)}")
    return out


# ------------------------------------------------------------------ 3. clusters
def clusters(df):
    g = df.groupby("activity").agg(
        TotalCO2=("co2", "sum"), AvgCO2=("co2", "mean"),
        TotalQuantity=("quantity", "sum"), Records=("co2", "size")).reset_index()
    g["CO2PerUnit"] = g["TotalCO2"] / g["TotalQuantity"].replace(0, np.nan)
    g["CO2PerUnit"] = g["CO2PerUnit"].fillna(0)
    k = min(3, len(g))
    if k < 2:
        g["ImpactCluster"] = "High Impact"; return g.rename(columns={"activity": "Activity"})
    X = StandardScaler().fit_transform(np.log1p(g[["TotalCO2", "CO2PerUnit"]]))
    g["cluster"] = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
    order = g.groupby("cluster")["TotalCO2"].mean().sort_values().index.tolist()
    names = ["Low Impact", "Medium Impact", "High Impact"] if k == 3 else ["Low Impact", "High Impact"]
    g["ImpactCluster"] = g["cluster"].map(dict(zip(order, names)))
    g["ImpactRank"] = g["ImpactCluster"].map({"Low Impact": 1, "Medium Impact": 2, "High Impact": 3})
    g = g.drop(columns="cluster").rename(columns={"activity": "Activity"})
    print(g.groupby("ImpactCluster")["Activity"].count().to_string())
    return g.sort_values("TotalCO2", ascending=False)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "ml_output"
    os.makedirs(outdir, exist_ok=True)
    data = load(src)
    forecast(data).to_csv(f"{outdir}/forecast_monthly.csv", index=False)
    anomalies(data).to_csv(f"{outdir}/anomalies.csv", index=False)
    clusters(data).to_csv(f"{outdir}/activity_clusters.csv", index=False)
    print(f"\nDone. Files written to: {os.path.abspath(outdir)}")
