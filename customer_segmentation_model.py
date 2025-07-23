#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import warnings
import os

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

os.makedirs("plots", exist_ok=True)


df = pd.read_csv("Mall_Customers.csv")
df.columns = [col.strip() for col in df.columns]

if 'Genre' in df.columns:
    df.rename(columns={'Genre': 'Gender'}, inplace=True)

le_gender = LabelEncoder()
df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
joblib.dump(le_gender, "label_encoder_gender.pkl")


def save_and_show_plot(fig, filename, title):
    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join("plots", filename)
    fig.savefig(path)
    plt.show()

fig = plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df)
save_and_show_plot(fig, "gender_distribution.png", "Gender Distribution")

fig = plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=15, kde=True)
save_and_show_plot(fig, "age_distribution.png", "Age Distribution")

fig = plt.figure(figsize=(6, 4))
sns.histplot(df['Annual Income (k$)'], bins=15, kde=True)
save_and_show_plot(fig, "income_distribution.png", "Annual Income Distribution")

fig = plt.figure(figsize=(6, 4))
sns.histplot(df['Spending Score (1-100)'], bins=15, kde=True)
save_and_show_plot(fig, "spending_distribution.png", "Spending Score Distribution")


features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler1.pkl')


inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

fig = plt.figure(figsize=(6, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('No. of Clusters')
plt.ylabel('Inertia')
save_and_show_plot(fig, "elbow_plot.png", "Elbow Plot")



kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters
joblib.dump(kmeans, 'kmeans_model1.pkl')



pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = pca_result[:, 0], pca_result[:, 1]

fig = plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
save_and_show_plot(fig, "cluster_pca.png", "Customer Segments (PCA)")



df.to_csv("clustered_customers11.csv", index=False)



def predict_cluster(age, income, score):
    loaded_scaler = joblib.load('scaler1.pkl')
    loaded_model = joblib.load('kmeans_model1.pkl')
    new_data = pd.DataFrame([[age, income, score]], columns=features)
    new_data_scaled = loaded_scaler.transform(new_data)
    cluster = loaded_model.predict(new_data_scaled)[0]
    return cluster


# In[ ]:




