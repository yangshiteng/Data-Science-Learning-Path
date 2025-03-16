## **Mean-Shift Clustering: Mathematical Understanding & Step-by-Step Implementation**

### **1. Introduction**
Mean-Shift is a **density-based clustering algorithm** that iteratively shifts data points towards regions of **higher density** in feature space. It does not require the number of clusters to be specified beforehand, unlike K-Means.

---

### **2. Mathematical Foundation of Mean-Shift**
#### **2.1. Kernel Density Estimation (KDE)**
Mean-Shift relies on **Kernel Density Estimation (KDE)** to estimate the probability density function (PDF) of the dataset. The density function at a point \( x \) is given by:

![image](https://github.com/user-attachments/assets/e105f682-a499-41e6-94d2-9b29f4fd4f8b)

---

#### **2.2. Mean Shift Vector Calculation**

![image](https://github.com/user-attachments/assets/59b220cb-4944-4f79-af0b-964080b92a5d)


---

### **3. Step by Step Python Implementation**

#### **Step 1: Install Dependencies**
If you don’t have `scikit-learn` installed, install it using:
```python
pip install scikit-learn matplotlib numpy
```

---

#### **Step 2: Import Required Libraries**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
```

---

#### **Step 3: Generate Sample Data**
We create a dataset with multiple clusters for testing.
```python
# Generate synthetic dataset with 3 clusters
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Scatter plot of raw data
plt.scatter(X[:, 0], X[:, 1], c='gray', marker='o', alpha=0.5)
plt.title("Generated Data for Clustering")
plt.show()
```

---

#### **Step 4: Estimate Bandwidth (Kernel Window Size)**
The **bandwidth** parameter defines the window size for mean-shift clustering.

```python
# Estimate optimal bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
print(f"Estimated Bandwidth: {bandwidth}")
```

---

#### **Step 5: Apply Mean-Shift Clustering**
```python
# Apply Mean-Shift Clustering
mean_shift = MeanShift(bandwidth=bandwidth)
mean_shift.fit(X)

# Get cluster centers and labels
cluster_centers = mean_shift.cluster_centers_
labels = mean_shift.labels_

# Count the number of clusters
num_clusters = len(np.unique(labels))
print(f"Number of clusters found: {num_clusters}")
```

---

#### **Step 6: Visualize Clustering Results**
```python
# Scatter plot of clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, marker='o')

# Mark cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label="Centers")

plt.title("Mean-Shift Clustering Results")
plt.legend()
plt.show()
```

---

#### **Step 7: New Data Points Clustering**

```python
# Define a new data point
new_point = np.array([[5, 0]])  # Example new data point

# Find the nearest cluster center using Euclidean distance
distances = cdist(new_point, cluster_centers)  # Compute distance to each cluster center
closest_cluster = np.argmin(distances)  # Get index of nearest cluster

print(f"New point {new_point} belongs to cluster {closest_cluster}")

import matplotlib.pyplot as plt

# Scatter plot of existing clusters
plt.scatter(X[:, 0], X[:, 1], c=mean_shift.labels_, cmap='viridis', alpha=0.5)

# Mark cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label="Centers")

# Plot the new point
plt.scatter(new_point[:, 0], new_point[:, 1], c='blue', marker='*', s=200, label="New Point")

plt.title("Mean-Shift Clustering with New Data Point")
plt.legend()
plt.show()

```





