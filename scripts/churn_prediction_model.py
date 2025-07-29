import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('cleaned_data.csv')

le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        df[column] = le.fit_transform(df[column])

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 2})

x = df.drop('Churn', axis=1)
y = df['Churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

joblib.dump(model, 'churn_model.pkl')
joblib.dump((x_test, y_test), "test_data.pkl")

print("Model trained and saved as 'churn_model.pkl'.")