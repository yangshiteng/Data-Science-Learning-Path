# **k-Nearest Neighbors (k-NN) for Outlier Detection: A Detailed Guide**

## **1. What is k-Nearest Neighbors (k-NN) for Outlier Detection?**
**k-Nearest Neighbors (k-NN)** is a **distance-based anomaly detection algorithm**. It identifies **outliers** by analyzing the distances between data points and their **k nearest neighbors**.

### **Key Concept**
- Normal data points **are close** to their neighbors.
- Outliers are **far away** from most other points.

### **Types of k-NN Outlier Detection**
1. **Distance-Based k-NN**: Uses the **distance to the k-th nearest neighbor** as an anomaly score.
2. **Average k-NN Distance**: Uses the **average distance to k neighbors** for anomaly detection.

---

## **2. How k-NN Works for Outlier Detection**
### **Step 1: Compute k-Nearest Neighbors Distance**
- For each data point, find the **k nearest neighbors**.
- Calculate the **distance** from the point to these k neighbors.

### **Step 2: Define an Outlier Score**
Two common ways:
1. **k-Distance Score** (Distance to the k-th neighbor).
2. **Mean k-Distance Score** (Average distance to k neighbors).

### **Step 3: Set an Outlier Threshold**
- Define a threshold (e.g., **95th percentile of k-distances**).
- Points with distances **above the threshold** are flagged as **outliers**.

---

## **3. Advantages and Limitations of k-NN for Outlier Detection**
| Feature | Advantages | Limitations |
|---------|------------|-------------|
| **Easy to Implement** | Simple and intuitive | Computationally expensive for large datasets |
| **No Assumptions on Data Distribution** | Works with any type of data | Choice of k affects results |
| **Works for Multidimensional Data** | Effective in multiple dimensions | Struggles with sparse data |
| **Distance-Based** | Detects both global and local anomalies | Sensitive to feature scaling |

---

## **4. Implementing k-NN for Outlier Detection in Python**
### **🔹 Step 1: Import Libraries**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
```

---

### **🔹 Step 2: Generate Sample Dataset**
We create a dataset with **normal points** and a few **outliers**.

```python
# Generate normal data (Gaussian distribution)
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=5, size=(100, 2))

# Generate outliers
outliers = np.array([[20, 20], [100, 100], [90, 110]])

# Combine dataset
data = np.vstack((normal_data, outliers))

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Feature1", "Feature2"])
```

---

### **🔹 Step 3: Compute k-Nearest Neighbors Distance**
```python
# Define k-NN model (using k=5 nearest neighbors)
k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(df[["Feature1", "Feature2"]])

# Compute distances to k nearest neighbors
distances, _ = nn.kneighbors(df[["Feature1", "Feature2"]])

# Compute k-th neighbor distance (outlier score)
df["k-Distance"] = distances[:, k-1]  # Distance to 5th nearest neighbor
```

---

### **🔹 Step 4: Set Outlier Threshold and Flag Outliers**
```python
# Set threshold as 95th percentile of k-distances
threshold = np.percentile(df["k-Distance"], 95)

# Identify outliers
df["Outlier"] = df["k-Distance"] > threshold

# Display detected outliers
print("Detected Outliers:\n", df[df["Outlier"]])
```

---

### **🔹 Step 5: Visualize Outliers**
```python
plt.figure(figsize=(8,6))
plt.scatter(df["Feature1"], df["Feature2"], c="blue", label="Normal Data")
plt.scatter(df[df["Outlier"]]["Feature1"], df[df["Outlier"]]["Feature2"], 
            c="red", label="Outliers", marker="o", s=100)
plt.title("Outlier Detection Using k-Nearest Neighbors (k-NN)")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.legend()
plt.show()
```

---

## **5. Explanation of the Implementation**
1. **Generate Dataset**:  
   - Create **100 normal points** (Gaussian distribution).  
   - Add **3 extreme outliers**.  

2. **Compute k-NN Distances**:
   - Fit a k-NN model with **k = 5**.
   - Compute **distance to 5th nearest neighbor**.

3. **Set Outlier Threshold**:
   - Use the **95th percentile of k-distances** as a threshold.
   - Flag points **above this threshold** as outliers.

4. **Plot the Data**:
   - Normal points are in **blue**.
   - Outliers are in **red**.

---

## **6. Choosing the Right k-NN Parameters**
| Parameter | Description | Suggested Values |
|-----------|-------------|------------------|
| `n_neighbors` | Number of neighbors for distance calculation | 3-10 (Higher for large datasets) |
| `metric` | Distance metric used | 'euclidean' (default), 'manhattan', 'cosine' |

---

## **7. When to Use k-NN for Outlier Detection**
✅ **Use k-NN when:**
- You need an **interpretable** distance-based anomaly detection method.
- Your dataset has **small to medium size**.
- Data is **not normally distributed**.

❌ **Avoid k-NN when:**
- The dataset is **very large** (high computational cost).
- The **choice of k** significantly impacts the results.
- Features are **not scaled properly** (sensitive to scale differences).

---

