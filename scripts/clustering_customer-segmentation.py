import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_data.csv')

features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
df = df[features].dropna()

#standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

#Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_features)
    inertia.append(km.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for optimal k')
plt.grid(True)
plt.savefig('elbow_plot.png')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

df['Cluster'] =clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Cluster', palette='Set2', data=df)
plt.title('Customer Segmentation by Monthly vs Total Charges')
plt.savefig('customer_segmentation.png')
plt.show() 

df.to_csv("clustered_customers_data.csv", index=False)
print("Clustered data saved to 'clustered_customers_data.csv'.")

