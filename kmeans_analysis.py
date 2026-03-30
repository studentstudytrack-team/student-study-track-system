# STEP 1️⃣ Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score 
import joblib

joblib.dump(model, "models/cluster_model.pkl")

# STEP 2️⃣ Load Your Dataset
df = pd.read_csv("study_behavior_dataset.csv")

# STEP 3️⃣ Select Numeric Attributes for Clustering
X = df[['Study_Time_Hours',
        'Attention_Span_Minutes',
        'Past_Score_Percentage',
        'Quiz_Score_Percentage']]

# STEP 4️⃣ Scale the Data (Important for KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 5️⃣ Apply KMeans
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# STEP 6️⃣ Check Silhouette Score
score = silhouette_score(X_scaled, df['Cluster'])
print("Silhouette Score:", round(score, 3))

# STEP 7️⃣ Plot Clusters (Using 2 attributes for visualization)
plt.figure(figsize=(8,6))

plt.scatter(df['Study_Time_Hours'],
            df['Quiz_Score_Percentage'],
            c=df['Cluster'],
            cmap='viridis')

# Plot Centroids (convert back to original scale)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(centroids[:,0],
            centroids[:,3],
            s=200,
            c='red',
            marker='X')

plt.xlabel("Study Time (Hours)")
plt.ylabel("Quiz Score (%)")
plt.title("K-Means Clustering (k=3)")
plt.show()