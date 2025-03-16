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

### **3. Step-by-Step Example**
Let's implement **Mean-Shift** from scratch to cluster a set of data points.

#### **Step 1: Import Libraries**
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
```

#### **Step 2: Generate Sample Data**
We create a simple dataset with two clusters.
```python
np.random.seed(42)

# Generate two clusters
cluster_1 = np.random.normal(loc=[3, 3], scale=1, size=(100, 2))
cluster_2 = np.random.normal(loc=[8, 8], scale=1, size=(100, 2))

# Combine into dataset
X = np.vstack((cluster_1, cluster_2))

# Scatter plot of raw data
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title("Generated Data")
plt.show()
```

#### **Step 3: Define Gaussian Kernel Function**
```python
def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
```

#### **Step 4: Implement Mean-Shift Algorithm**
```python
def mean_shift(X, bandwidth=1.0, max_iter=100, tol=1e-3):
    # Initialize points to their original positions
    points = np.copy(X)
    num_points = points.shape[0]

    for _ in range(max_iter):
        new_points = np.zeros_like(points)
        
        for i in range(num_points):
            distances = np.linalg.norm(points - points[i], axis=1)
            weights = gaussian_kernel(distances, bandwidth)

            # Compute weighted mean shift
            new_points[i] = np.sum(weights[:, np.newaxis] * points, axis=0) / np.sum(weights)

        # Check for convergence
        shift_distance = np.linalg.norm(new_points - points, axis=1)
        if np.max(shift_distance) < tol:
            break
        
        points = new_points  # Update positions

    return points
```

#### **Step 5: Apply Mean-Shift Clustering**
```python
# Run Mean-Shift Clustering
bandwidth = 2.0  # Set the window size
shifted_points = mean_shift(X, bandwidth)

# Get unique cluster centers
cluster_centers = np.unique(np.round(shifted_points, decimals=2), axis=0)

# Assign clusters based on proximity
labels = np.argmin(cdist(X, cluster_centers), axis=1)
```

#### **Step 6: Visualize the Clustering Results**
```python
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label="Centers")
plt.title("Mean-Shift Clustering Results")
plt.legend()
plt.show()
```

---

### **4. Key Takeaways**
- **Non-parametric**: Does not require specifying the number of clusters beforehand.
- **Density-based**: Finds natural cluster centers by shifting towards denser regions.
- **Bandwidth Selection**: The choice of \( h \) affects cluster formation.
- **Computationally Expensive**: Slower than K-Means, especially for large datasets.

Would you like an interactive visualization or parameter tuning guide? 🚀
