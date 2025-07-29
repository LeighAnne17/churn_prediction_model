import pandas as pd

#load data
df = pd.read_csv("mock_telco_churn_data.csv")
print(df.head())

#checking nulls
print(df.isnull().sum())

#data types
print(df.dtypes)


#Data Cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
                         
def tenure_group(tenure):
    if tenure < 12:
        return "< 1 year"
    elif tenure < 24:
        return "1-2 years"
    elif tenure < 48:
        return "2-4 years"
    elif tenure < 60:
        return "4-5 years"
    else:
        return "5+ years"
    
df["TenureGroup"] = df["Tenure"].apply(tenure_group)

df.to_csv("cleaned_data.csv", index=False)
print("File saved successfully.")