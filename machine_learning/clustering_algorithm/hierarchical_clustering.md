# **Hierarchical Clustering: A Step-by-Step Mathematical Explanation**

Hierarchical Clustering is a **tree-based clustering algorithm** that groups data into a **hierarchical structure** using a **dendrogram**.

## **1. Types of Hierarchical Clustering**
There are two types:
1. **Agglomerative Hierarchical Clustering (AHC) (Bottom-Up)** → **Start with each point as its own cluster and merge until one big cluster remains.**
2. **Divisive Hierarchical Clustering (Top-Down)** → **Start with one big cluster and split into smaller clusters recursively.**

### **Agglomerative Hierarchical Clustering (AHC) is more commonly used.**

---

## **2. Steps of Agglomerative Hierarchical Clustering**
Given a dataset with `N` data points, AHC follows these steps:

### **Step 1: Compute Distance Matrix**
- Calculate the **Euclidean distance** between every pair of points.

### **Step 2: Merge the Closest Clusters**
- Find the **two closest clusters** based on a **linkage criterion**.
- Merge them into a **new cluster**.

### **Step 3: Update the Distance Matrix**
- Compute the distance between the **new cluster** and the remaining clusters.
- Repeat Steps **2 & 3** until **only one cluster remains**.

### **Step 4: Create the Dendrogram**
- The **merging process is recorded** in a tree-like diagram called a **dendrogram**.
- **Cutting the dendrogram at a certain height** determines the number of clusters.

---

## **3. Step-by-Step Example with Mathematical Calculation**
We will perform **Agglomerative Hierarchical Clustering** step-by-step.

### **Given 5 Data Points**
| Point | X  | Y  |
|--------|----|----|
| **A** | 1  | 2  |
| **B** | 2  | 3  |
| **C** | 3  | 3  |
| **D** | 5  | 7  |
| **E** | 6  | 8  |

---

### **Step 1: Compute Pairwise Euclidean Distance**
The **Euclidean distance formula**:

\[
d(A, B) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
\]

Let's compute all pairwise distances:

| From → To  | A  | B  | C  | D  | E  |
|------------|----|----|----|----|----|
| **A (1,2)** | 0  | 1.41 | 2.24 | 7.21 | 8.49 |
| **B (2,3)** | 1.41 | 0  | 1.00 | 5.83 | 7.07 |
| **C (3,3)** | 2.24 | 1.00 | 0  | 5.00 | 6.40 |
| **D (5,7)** | 7.21 | 5.83 | 5.00 | 0  | 1.41 |
| **E (6,8)** | 8.49 | 7.07 | 6.40 | 1.41 | 0  |

✅ The closest pair is **(B, C) with distance 1.00**.

---

### **Step 2: Merge the Closest Clusters**
We **merge B and C** into a new cluster **BC**.

New cluster positions:
- **BC = Mean(B, C)** → \(\left(\frac{2+3}{2}, \frac{3+3}{2}\right) = (2.5, 3)\)

---

### **Step 3: Update the Distance Matrix**
We update the distance matrix using **single linkage** (minimum distance):

| From → To  | A  | BC | D  | E  |
|------------|----|----|----|----|
| **A (1,2)** | 0  | 1.58 | 7.21 | 8.49 |
| **BC (2.5,3)** | 1.58 | 0  | 5.39 | 6.82 |
| **D (5,7)** | 7.21 | 5.39 | 0  | 1.41 |
| **E (6,8)** | 8.49 | 6.82 | 1.41 | 0  |

✅ The closest pair is **(D, E) with distance 1.41**.

---

### **Step 4: Merge the Next Closest Clusters**
- **Merge D and E** into a new cluster **DE**.
- **DE = Mean(D, E)** → \(\left(\frac{5+6}{2}, \frac{7+8}{2}\right) = (5.5, 7.5)\)

---

### **Step 5: Update the Distance Matrix**
| From → To  | A  | BC | DE  |
|------------|----|----|----|
| **A (1,2)** | 0  | 1.58 | 7.86 |
| **BC (2.5,3)** | 1.58 | 0  | 5.70 |
| **DE (5.5,7.5)** | 7.86 | 5.70 | 0  |

✅ The closest pair is **(A, BC) with distance 1.58**.

---

### **Step 6: Merge A and BC**
- **Merge A and BC into ABC**.
- **ABC = Mean(A, BC)** → \(\left(\frac{1+2.5}{2}, \frac{2+3}{2}\right) = (1.75, 2.5)\)

---

### **Step 7: Final Merge**
| From → To  | ABC | DE |
|------------|----|----|
| **ABC (1.75,2.5)** | 0  | 6.78 |
| **DE (5.5,7.5)** | 6.78 | 0  |

✅ The closest pair is **(ABC, DE)**. **Final merge** results in a **single cluster**.

---

### **Final Clustering Representation**
The hierarchical tree structure:

```
          ABCDE
         /     \
       ABC     DE
      /   \    /  \
     A    BC  D    E
         /   \
        B     C

```

---

## **4. Types of Linkage Criteria**
The way distances between clusters are updated affects the final clustering.

| Linkage Type | How Distance is Measured | Effect |
|-------------|------------------------|--------|
| **Single Linkage** | Distance between the closest points in clusters | Tends to form long chains |
| **Complete Linkage** | Distance between the farthest points | Produces compact clusters |
| **Average Linkage** | Average distance between all points | Balanced clustering |
| **Centroid Linkage** | Distance between cluster centroids | Sensitive to outliers |

---

## **5. Key Takeaways**
✅ **Hierarchical Clustering creates a dendrogram**, showing the **merging process**.  
✅ **No need to predefine `K`** (the number of clusters).  
✅ **Different linkage methods** affect results.  
✅ **Computationally expensive for large datasets** (\(O(n^3)\) complexity).  

