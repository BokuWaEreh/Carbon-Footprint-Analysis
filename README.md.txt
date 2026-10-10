#Carbon Footprint Analysis

#Project Overview
Carbon Footprint Analysis is a data analytics project that explores carbon-emission data through an interactive Power BI dashboard and machine-learning techniques. It aims to show emission trends, forecast future emissions, flag potentially unusual records, and group activities by their emission impact.
Objectives
Analyze carbon-emission data.
Explore emission trends in Power BI.
Forecast emissions for the next six months.
Flag potentially unusual emission records using Isolation Forest.
Group activities by emission impact using K-Means clustering.

#Technologies Used
Python
Pandas and NumPy
Scikit-learn
Statsmodels
Power BI
CSV/Excel-compatible data files
Machine Learning Methods
Holt-Winters Exponential Smoothing — forecasts monthly emissions.
Isolation Forest — flags records that differ from typical patterns; a flag does not automatically mean the record is incorrect.
K-Means Clustering — groups activities using emission-related features.

#Project Files
`ml_insights.py` — Python script for the machine-learning analysis.
`carbon_data` — csv file for dataset
`carbon_footprint_ML` — Power BI report 
`anomalies, forecast, activity_cluster` — csv files for activities, forecast, anomalies.
`outputs` — pdf file for output images

#Disclaimer
This project is intended for learning and analytical demonstration. 
