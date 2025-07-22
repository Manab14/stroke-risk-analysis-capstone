# 🧠 Stroke Risk Analysis – Capstone Project
# 📌 Project Overview
This project focuses on analysing a healthcare dataset to identify key factors associated with stroke risk. The analysis combines Python, SQL, Excel, and Power BI to perform end-to-end data analytics, build predictive models, and develop an interactive dashboard for clear healthcare insights.

# 🔍 Objectives
Clean and preprocess stroke health data efficiently.

Perform univariate and bivariate exploratory data analysis (EDA).

Conduct feature engineering to prepare data for modelling.

Build a Logistic Regression model to predict stroke occurrence.

Visualize critical risk factors and patterns through a Power BI dashboard.

# 🛠️ Tools & Technologies Used
Python: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

SQL: Views and aggregation queries for targeted analysis

Excel: Quick data profiling and feature adjustments

Power BI: Dashboard design and interactive visualization

# 📂 Project Workflow
Data Cleaning (Python)

Handled missing values (median imputation for BMI).

Encoded categorical variables.

Converted data types as required.

Exploratory Data Analysis

Univariate analysis to understand distributions.

Bivariate analysis using groupby aggregations and plots for risk factors like hypertension, age, and smoking status.

Feature Engineering

Created age groups for better interpretability.

Engineered BMI categories (underweight, normal, overweight, obese).

Used Excel for quick feature checks and category binning.

SQL Analysis

Created views for stroke by gender, average BMI by work type, stroke rates by age group, and hypertension-stroke relationship.

Queried KPIs such as total stroke cases and stroke prevalence rate.

Machine Learning Model

Logistic Regression built with Scikit-learn.

Achieved ~80% accuracy.

Evaluated using confusion matrix and classification report.

Power BI Dashboard

Integrated SQL views and dataset.

Designed interactive visuals: stroke by gender, age group, BMI category, smoking status, and more.

Created key KPI cards for quick executive insights.

# 📊 Key Findings
Hypertension and heart disease significantly increase stroke risk.

Stroke prevalence is higher among older age groups (especially 60+).

Average BMI and glucose levels show strong association with stroke occurrence.


# 💡 Future Improvements
Incorporate advanced balancing techniques (e.g. SMOTE) for imbalanced data.

Experiment with ensemble models for improved predictive performance.

Integrate real-time data APIs for live healthcare dashboards.
