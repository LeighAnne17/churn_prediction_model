import pandas as pd
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})

df_encoded['gender'] = df_encoded['gender'].map({'Male': 1, 'Female': 0})
df_encoded = pd.get_dummies(df_encoded, drop_first=True)

from sklearn.model_selection import train_test_split

x = df_encoded.drop('Churn', axis=1)
x = df_encoded['Churn']

x_train, x_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
