import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv('cleaned_data.csv')


model = joblib.load("churn_model.pkl")
x_test, y_test = joblib.load("test_data.pkl")

y_test = y_test.replace({2: 0})

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

#ROC auc score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC AUC Score: {roc_auc:.2f}")

#ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='blue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()