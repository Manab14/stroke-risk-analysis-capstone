import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load dataset
df=pd.read_csv("C:\\Users\\MANAB\\Downloads\\archive (2)\\healthcare-dataset-stroke-data.csv")
print(df.info())
print(df.describe())
print(df.isnull().sum())

print("Average BMI: ",df["bmi"].mean())

# Impute missing BMI values with median (robust against outliers)
df["bmi"].fillna(df["bmi"].median(),inplace=True)
print(df.isnull().sum())

# Save cleaned data
df.to_csv("Cleaned Stroke Data.csv",index=False)

# Encoding gender for ML usage
df["labeled_gender"]=df["gender"].replace({"Male":0,"Female":1,"Others":2})
print(df["labeled_gender"])

# Histogram: Age distribution
sns.histplot(df["age"],kde=True,color='teal')
plt.show()
# Insight: Age distribution is right-skewed with more observations in higher age groups, indicating older population dominance in dataset.

# Histogram: BMI distribution
sns.histplot(df["bmi"],kde=True,color='teal')
plt.show()
# Insight: BMI is normally distributed with slight right skew; majority BMI is between 20-35 indicating overweight tendency.

# Histogram: Average Glucose Level distribution
sns.histplot(df["avg_glucose_level"],kde=True,color='teal')
plt.show()
# Insight: Avg glucose level is highly right-skewed; many people have normal levels but a significant tail of high glucose exists (diabetic risk).

# Countplot: Gender distribution
sns.countplot(df["gender"],palette="coolwarm")
plt.show()
# Insight: Slightly more females than males in dataset; very few 'Other' category entries.

# Countplot: Work type distribution
sns.countplot(df["work_type"],palette="coolwarm")
plt.show()
# Insight: Majority are Private sector employees, followed by Self-employed and Govt jobs; Children and Never worked are smaller groups.

# Countplot: Smoking status distribution
sns.countplot(df["smoking_status"],palette="coolwarm")
plt.show()
# Insight: Many entries are 'never smoked'; fewer 'formerly smoked' and 'smokes' categories.

stroke_rate_by_gender=df.groupby("gender")["stroke"].mean()
print(stroke_rate_by_gender)

sns.barplot(stroke_rate_by_gender,palette="coolwarm")
plt.show()
# Insight: Stroke rate is slightly higher in females compared to males in this dataset.

stroke_count_by_work_type=df.groupby("work_type")["stroke"].sum()
print(stroke_count_by_work_type)

sns.barplot(stroke_count_by_work_type,palette="coolwarm")
plt.show()
# Insight: Private sector employees have the highest absolute stroke counts, indicating occupational prevalence due to sample size.

stroke_count_by_smoking_status=df.groupby("smoking_status")["stroke"].sum()
print(stroke_count_by_smoking_status)

sns.barplot(stroke_count_by_smoking_status,palette="coolwarm")
plt.show()
# Insight: Highest stroke counts in 'never smoked' category; however, this needs normalization as group sizes differ.

stroke_rate_by_marraige=df.groupby("ever_married")["stroke"].mean()
print(stroke_rate_by_marraige)

sns.barplot(stroke_rate_by_marraige,palette="coolwarm")
plt.show()
# Insight: Married individuals have a significantly higher stroke rate, possibly due to age confounding (married people are older).

stroke_rate_by_residence=df.groupby("Residence_type")["stroke"].mean()
print(stroke_rate_by_residence)

sns.barplot(stroke_rate_by_residence,palette="coolwarm")
plt.show()
# Insight: Urban and rural stroke rates are similar, suggesting residence type alone isn’t a strong differentiator in this dataset.

sns.boxplot(x="stroke",y="age",data=df,palette="coolwarm")
plt.show()
# Insight: Stroke patients are generally older; median age for stroke=1 is higher than stroke=0 group.

sns.boxplot(x="stroke",y="bmi",data=df,palette="coolwarm")
plt.show()
# Insight: BMI distribution is similar across stroke groups, indicating BMI alone may not be a strong predictor.

sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm")
plt.show()
# Insight: Age, avg_glucose_level, and hypertension show positive correlations with stroke; no strong multicollinearity observed.

# Save modified data
df.to_csv("Modified Stroke Data.csv",index=False)

# Reload modified data for modelling
sdf=pd.read_csv("Modified Stroke Data.csv")
print(sdf.head())
print(sdf.isnull().sum())

# Feature selection for model
x = sdf.select_dtypes(include='number')
X=x.drop(["id","stroke"],axis=1)
print(X.isnull().sum())
y=sdf["stroke"]

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standard scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# Logistic Regression model
logreg=LogisticRegression(max_iter=1000,class_weight="balanced")
logreg.fit(X_train_scaled,y_train)

# Predictions
y_pred=logreg.predict(X_test_scaled)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))
# Insight: The model achieves good recall for stroke detection due to class_weight balancing; precision is low due to dataset imbalance.

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Insight: Many true negatives, moderate true positives; false positives remain low.

y_prob = logreg.predict_proba(X_test_scaled)[:,1]
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
# Insight: ROC-AUC ~0.8 indicates decent discriminatory power of logistic regression for stroke prediction.

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logreg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print(coefficients)
# Insight: Features with high positive coefficients contribute more to stroke risk (e.g. avg_glucose_level, age), while negative coefficients reduce the likelihood.

