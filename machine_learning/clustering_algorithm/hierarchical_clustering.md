### Introduction to Hierarchical Clustering

Hierarchical Clustering is a type of unsupervised machine learning algorithm used to group similar data points into clusters. Unlike K-Means clustering, which requires the number of clusters to be specified in advance, hierarchical clustering builds a hierarchy of clusters, either in a bottom-up (agglomerative) or top-down (divisive) approach. The most common approach is **agglomerative hierarchical clustering**, which we will focus on.

### Agglomerative Hierarchical Clustering: Step-by-Step

Agglomerative hierarchical clustering starts by treating each data point as a single cluster. Then, it iteratively merges the closest pairs of clusters until all data points are in a single cluster or a stopping criterion is met.

#### Step 1: Define the Dataset
Let’s assume we have the following dataset with 5 data points in 2D space:

| Point | X   | Y   |
|-------|-----|-----|
| A     | 1.0 | 1.0 |
| B     | 1.5 | 1.5 |
| C     | 5.0 | 5.0 |
| D     | 5.5 | 5.5 |
| E     | 3.0 | 4.0 |

#### Step 2: Compute the Distance Matrix
The first step is to compute the pairwise distance between all data points. We will use the **Euclidean distance** as the distance metric.

![image](https://github.com/user-attachments/assets/810f8345-ac5b-4582-aa1f-3ece17e5f6ce)

The distance matrix for the dataset is:

|       | A     | B     | C     | D     | E     |
|-------|-------|-------|-------|-------|-------|
| **A** | 0.0   | 0.71  | 5.66  | 6.36  | 3.61  |
| **B** | 0.71  | 0.0   | 4.95  | 5.66  | 2.92  |
| **C** | 5.66  | 4.95  | 0.0   | 0.71  | 2.24  |
| **D** | 6.36  | 5.66  | 0.71  | 0.0   | 2.92  |
| **E** | 3.61  | 2.92  | 2.24  | 2.92  | 0.0   |

#### Step 3: Merge the Closest Clusters
The smallest distance in the matrix is between **A** and **B** (0.71). We merge these two points into a new cluster **AB**.

#### Step 4: Update the Distance Matrix
After merging **A** and **B**, we need to update the distance matrix. We use the **single linkage** method, which defines the distance between two clusters as the shortest distance between any single pair of points from the two clusters.

The new distance matrix is:

|       | AB    | C     | D     | E     |
|-------|-------|-------|-------|-------|
| **AB**| 0.0   | 4.95  | 5.66  | 2.92  |
| **C** | 4.95  | 0.0   | 0.71  | 2.24  |
| **D** | 5.66  | 0.71  | 0.0   | 2.92  |
| **E** | 2.92  | 2.24  | 2.92  | 0.0   |

#### Step 5: Merge the Next Closest Clusters
The smallest distance in the updated matrix is between **C** and **D** (0.71). We merge these two points into a new cluster **CD**.

#### Step 6: Update the Distance Matrix Again
Using single linkage, the new distance matrix is:

|       | AB    | CD    | E     |
|-------|-------|-------|-------|
| **AB**| 0.0   | 4.95  | 2.92  |
| **CD**| 4.95  | 0.0   | 2.24  |
| **E** | 2.92  | 2.24  | 0.0   |

#### Step 7: Merge the Next Closest Clusters
The smallest distance is now between **CD** and **E** (2.24). We merge these clusters into a new cluster **CDE**.

#### Step 8: Update the Distance Matrix
The updated distance matrix is:

|       | AB    | CDE   |
|-------|-------|-------|
| **AB**| 0.0   | 2.92  |
| **CDE**| 2.92  | 0.0   |

#### Step 9: Merge the Final Clusters
The final step is to merge **AB** and **CDE** into a single cluster **ABCDE**.

#### Step 10: Visualize the Dendrogram
The hierarchical clustering process can be visualized using a **dendrogram**, which shows the order and distances of the merges.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Define the dataset
data = np.array([
    [1.0, 1.0],  # A
    [1.5, 1.5],  # B
    [5.0, 5.0],  # C
    [5.5, 5.5],  # D
    [3.0, 4.0]   # E
])

# Step 2: Perform hierarchical clustering using single linkage
# The 'linkage' function computes the hierarchical clustering
# 'method="single"' specifies single linkage
Z = linkage(data, method='single')

# Step 3: Plot the dendrogram
plt.figure(figsize=(8, 5))
plt.title("Dendrogram for Agglomerative Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
dendrogram(Z, labels=['A', 'B', 'C', 'D', 'E'])
plt.show()
```

![image](https://github.com/user-attachments/assets/977cf883-4935-4858-a945-809a07548719)

![image](https://github.com/user-attachments/assets/26528245-ef30-4d63-9017-731f928e754f)

### Summary of Steps:
1. Start with each data point as a single cluster.
2. Compute the distance matrix.
3. Merge the two closest clusters.
4. Update the distance matrix.
5. Repeat steps 3-4 until all data points are in a single cluster.

### Key Points:
- **Distance Metric**: Common metrics include Euclidean, Manhattan, and Cosine similarity.
- **Linkage Criteria**: Determines how the distance between clusters is calculated. Common methods include single linkage, complete linkage, and average linkage.
- **Dendrogram**: A tree-like diagram that records the sequences of merges or splits.

### Applications:
- **Biology**: Phylogenetic tree construction.
- **Social Network Analysis**: Community detection.
- **Image Processing**: Image segmentation.

