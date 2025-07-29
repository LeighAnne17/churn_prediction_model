import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_data.csv")

sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.savefig("churn_distribution.png")
plt.show()

#Contract type
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.xticks(rotation=45)
plt.savefig("contract type.png")
plt.show()

#Tenure group
sns.countplot(x="TenureGroup", hue="Churn", data=df, order=["< 1 year", "1-2 years", "2-4 years", "4-5 years", "5+ years"])
plt.title("Churn by Tenure Group")
plt.savefig("tenure.png")
plt.show()

#Monthly charges
sns.violinplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn: ")
plt.savefig("monthly_charges.png")
plt.show()

df["TotalCharges"] = df["TotalCharges"].astype(str)
df = df[df["TotalCharges"].str.replace('.', '', 1).str.isnumeric()]
df["TotalCharges"] = df["TotalCharges"].astype(float)

#Correlation
numeric_cols = ["SeniorCitizen", "Tenure", "MonthlyCharges", "TotalCharges"]
print(df[numeric_cols].dtypes)
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(subset=numeric_cols)

correlation = df[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

