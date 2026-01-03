import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
def load_ecg_data(file_path):
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=[np.number]) 
    return df.values

file_path = r"D:\Features.csv"
X = load_ecg_data(file_path)
X = StandardScaler().fit_transform(X) 

inertias = []
silhouette_scores = []
k_range = range(2, 11)  # Testing k values from 2 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    
    # Silhouette Score calculation
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X, labels))

# Plot Elbow Method results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

optimal_k = 3  # Choose based on elbow plot and silhouette scores
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_

# Final Silhouette Score
silhouette_avg = silhouette_score(X, labels)
print(f"\nFinal Silhouette Score for k={optimal_k}: {silhouette_avg:.4f}")

# Plot Clusters (using first two features for visualization)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')
plt.xlabel('Standardized Feature 1')
plt.ylabel('Standardized Feature 2')
plt.title(f'ECG Feature Clustering (k={optimal_k})')
plt.legend()
plt.show()
