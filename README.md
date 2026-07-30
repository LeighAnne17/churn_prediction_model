#  Customer Churn Prediction & Customer Segmentation
##  Project Overview
Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing existing customers is often more expensive than acquiring new ones, making early churn prediction an important business objective.

This project applies **data analytics**, **machine learning** and **customer segmentation** techniques to identify customers who are likely to churn and group customers with similar behavioural patterns. The project follows a complete data science workflow from data cleaning to model evaluation and business insights.

---

# 🎯 Objectives

* Clean and prepare customer data for analysis
* Explore factors influencing customer churn
* Build a machine learning model to predict churn
* Evaluate model performance using industry-standard metrics
* Segment customers using K-Means clustering
* Generate actionable business insights to support customer retention strategies

---

# Dataset
The project uses a telecommunications customer dataset containing customer demographic information, service subscriptions, billing information and churn status

### Features include:
* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Internet Service
* Contract Type
* Payment Method
* Monthly Charges
* Total Charges
* Churn Status

---

# 🛠 Technologies Used
### Programming
* Python

### Libraries
* pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

### Development Environment
* Visual Studio Code
* Jupyter Notebook

---

# Project Structure

```text
Customer-Churn-Prediction/

│
├── data/
│   ├── mock_telco_churn_data.csv
│   ├── cleaned_data.csv
│
├── scripts/
│   ├── data-cleaning_and-exploring.py
│   ├── data-analysis.py
│   ├── churn_prediction_model.py
│   ├── churn_eval.py
│   └── clustering_customer-segmentation.py
│
├── visuals/
│   ├── churn_distribution.png
│   ├── contract type.png
│   ├── tenure.png
│   ├── monthly_charges.png
│   ├── correlation_heatmap.png
│   ├── roc_curve.png
│   ├── elbow_plot.png
│   └── customer_segmentation.png
│
├── churn_model.pkl
├── requirements.txt
└── README.md
```

---

#  Exploratory Data Analysis

Several visualizations were created to understand customer behavior and identify factors associated with churn.
The analysis includes:

* Customer churn distribution
* Contract type analysis
* Tenure group analysis
* Monthly charges comparison
* Correlation heatmap
* Customer segmentation visualization

These visualizations helped identify important trends before building the predictive model.
### Customer Churn Distribution
This chart shows the overall distribution of customers who stayed versus those who churned

<img width="640" height="480" alt="churn_distribution" src="https://github.com/user-attachments/assets/109d349b-fafc-4556-9cfe-fc98b45d2615" />


### Churn by Contract Type
Customers on month-to-month contracts exhibited noticeably higher churn rates compared to customers with longer-term contracts

<img width="640" height="480" alt="contract type" src="https://github.com/user-attachments/assets/e815a601-77ba-4731-93a9-2d4a461b6f37" />

### Churn by Customer Tenure
Customers with shorter tenures were significantly more likely to leave, while long-term customers showed greater loyalty

<img width="640" height="480" alt="tenure" src="https://github.com/user-attachments/assets/a191f287-fa02-4ce3-923a-2587360dc11f" />


### Monthly Charges Distribution
Monthly charges were compared across churned and retained customers to determine whether pricing influenced customer retention

<img width="640" height="480" alt="monthly_charges" src="https://github.com/user-attachments/assets/c8a72c9e-8542-4d5d-a95a-728a9354d9ac" />

### Correlation Heatmap
The correlation matrix highlights relationships between the numerical variables used throughout the project

<img width="640" height="480" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/5ccd099d-1385-45c9-a70f-6713f455d95d" />


### Customer Segmentation
Customers were grouped into clusters based on spending and subscription characteristics



---

#  Machine Learning Model
The project uses a **Random Forest Classifier** to predict customer churn.

### Workflow

* Data Cleaning
* Feature Engineering
* Label Encoding
* Train/Test Split
* Random Forest Model Training
* Model Evaluation
* Customer Segmentation

---

# Model Evaluation
The model achieved a baseline ROC-AUC score of 0.64, indicating that it successfully learned meaningful customer churn patterns from the available features
<img width="800" height="600" alt="roc_curve" src="https://github.com/user-attachments/assets/9d4b72d4-5c2c-436a-854f-5b08fce20886" />


---

# Model Performance
| Metric        | Result                |
| ------------- | --------------------- |
| Algorithm     | Random Forest         |
| ROC-AUC Score | **0.64**              |
| Evaluation    | Classification Report |
| ROC Curve     | Generated             |

Although the dataset is relatively small, the model successfully learned meaningful customer churn patterns and provided a realistic baseline for future improvements

---

# Customer Segmentation
K-Means clustering was applied using customer tenure, monthly charges and total charges

The segmentation identifies customer groups with similar spending and subscription behaviour, providing useful information for targeted marketing campaigns and retention strategies

---

# Business Insights

The analysis suggests several factors contribute to customer churn:
* Customers on month-to-month contracts are more likely to churn
* Customers with shorter tenure show higher churn rates
* Monthly charges influence customer retention
* Customer segmentation can help businesses develop personalised retention campaigns

These insights can support data-driven business decisions aimed at reducing customer churn

---

# ▶️ How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project in the following order:

```bash
python scripts/data-cleaning_and-exploring.py

python scripts/data-analysis.py

python scripts/churn_prediction_model.py

python scripts/churn_eval.py

python scripts/clustering_customer-segmentation.py
```

---

