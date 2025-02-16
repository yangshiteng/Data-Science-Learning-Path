# **K-Means Clustering: A Detailed Tutorial with Python Implementation**

## **1. Introduction to K-Means Clustering**
K-Means is a **centroid-based clustering algorithm** that partitions data into **K clusters**. It minimizes the variance within clusters by iteratively updating cluster centroids.

### **How K-Means Works**
1. **Choose the number of clusters (K)**.
2. **Randomly initialize K cluster centroids**.
3. **Assign each data point to the nearest centroid**.
4. **Update centroids** by taking the mean of points in each cluster.
5. **Repeat steps 3 and 4** until centroids do not change significantly.

---

## **2. Steps in K-Means Algorithm**
### **Step 1: Choose Number of Clusters (K)**
- The **Elbow Method** or **Silhouette Score** is often used to select an optimal **K**.

### **Step 2: Initialize Cluster Centroids**
- Centroids are initialized randomly or using **K-Means++** (better initialization technique).

### **Step 3: Assign Data Points to Nearest Centroid**
- Calculate **Euclidean distance** between each data point and the centroids.
- Assign points to the **closest** centroid.

### **Step 4: Update Centroids**
- Compute the **mean of all points** assigned to each cluster.
- Set new centroids as the mean values.

### **Step 5: Repeat Until Convergence**
- Continue **re-assigning points and updating centroids** until centroids **no longer change**.

---

## **3. Implementing K-Means Clustering in Python**
### **🔹 Step 1: Import Libraries**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```

---

### **🔹 Step 2: Generate Sample Data**
We create a dataset with **three clusters**.

```python
# Generate synthetic dataset with 3 clusters
np.random.seed(42)
X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to DataFrame
df = pd.DataFrame(X_scaled, columns=["Feature1", "Feature2"])
```

---

### **🔹 Step 3: Use the Elbow Method to Find Optimal K**
The **Elbow Method** helps determine the best value of **K** by plotting **inertia** (sum of squared distances from points to their cluster center).

```python
# Compute K-Means for different K values
inertia = []
K_values = range(1, 11)

for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)  # Inertia = sum of squared distances

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K_values, inertia, marker="o", linestyle="--")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()
```
🔹 **Interpretation**: The optimal **K** is at the "elbow" of the curve.

---

### **🔹 Step 4: Run K-Means Clustering**
```python
# Apply K-Means with the chosen K (e.g., K=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Get cluster centroids
centroids = kmeans.cluster_centers_
```

---

### **🔹 Step 5: Visualize Clusters**
```python
# Plot Clusters
plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    cluster_data = df[df["Cluster"] == cluster]
    plt.scatter(cluster_data["Feature1"], cluster_data["Feature2"], label=f"Cluster {cluster}")

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, color="black", label="Centroids")

plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("K-Means Clustering")
plt.legend()
plt.show()
```

---

### **🔹 Step 6: Evaluate Clustering Using Silhouette Score**
Silhouette Score measures how well samples are clustered (higher score = better clustering).

```python
sil_score = silhouette_score(X_scaled, df["Cluster"])
print(f"Silhouette Score: {sil_score:.3f}")
```
🔹 **Interpretation**: A score close to **1.0** indicates well-defined clusters.

---

## **4. K-Means Hyperparameters**
| Parameter | Description | Recommended Values |
|-----------|-------------|------------------|
| `n_clusters` | Number of clusters | Use Elbow Method |
| `init` | Initialization method (`random`, `k-means++`) | Default: `k-means++` |
| `n_init` | Number of times K-Means runs with different centroid seeds | 10 (default) |
| `max_iter` | Maximum number of iterations | 300 (default) |
| `tol` | Convergence threshold | 1e-4 |

---

## **5. When to Use K-Means**
✅ **Best for:**
- Large datasets where **quick and efficient clustering** is needed.
- Well-separated clusters (e.g., customer segmentation).
- Applications in **image compression, anomaly detection**.

❌ **Avoid if:**
- Data contains **overlapping clusters**.
- Clusters are **not spherical** (try **DBSCAN** instead).
- Outliers significantly affect the clustering.

---
