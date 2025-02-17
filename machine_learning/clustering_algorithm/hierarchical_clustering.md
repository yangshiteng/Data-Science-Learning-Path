# **Hierarchical Clustering: A Comprehensive Guide with Correct Distance Similarity Levels**  

## **1. Introduction to Hierarchical Clustering**  
Hierarchical clustering is a **tree-based clustering algorithm** that groups data into a **nested hierarchy of clusters**. Unlike K-Means, it does **not require specifying the number of clusters (K) in advance**. Instead, a **dendrogram** is generated to help determine the optimal number of clusters.

---

## **2. Types of Hierarchical Clustering**
1. **Agglomerative Hierarchical Clustering (AHC) (Bottom-Up Approach)**  
   - **Each data point starts as its own cluster.**  
   - **Clusters are iteratively merged** based on distance similarity until one final cluster remains.  
   - **Most commonly used method** in hierarchical clustering.

2. **Divisive Hierarchical Clustering (Top-Down Approach)**  
   - **All data points start in one large cluster.**  
   - **Clusters are recursively split** until each point forms its own cluster.  
   - **Less commonly used due to high computational cost**.

We will focus on **Agglomerative Hierarchical Clustering (AHC)**.

![image](https://github.com/user-attachments/assets/d31b1986-6371-4540-9902-3081f98e40c7)

---

## **3. Steps of Agglomerative Hierarchical Clustering**
Given 5 points:

| Point | X  | Y  |
|--------|----|----|
| **A** | 1  | 2  |
| **B** | 2  | 2  |
| **C** | 2  | 3  |
| **D** | 5  | 7  |
| **E** | 6  | 8  |

---

### **Step 1: Compute Pairwise Distance Matrix**
Using the **Euclidean distance formula**:

![image](https://github.com/user-attachments/assets/13dd1c26-0f01-495c-8830-3f72df551dba)

| From → To  | A  | B  | C  | D  | E  |
|------------|----|----|----|----|----|
| **A (1,2)** | 0  | 1.00 | 1.41 | 7.21 | 8.49 |
| **B (2,2)** | 1.00 | 0  | 1.00 | 5.83 | 7.07 |
| **C (2,3)** | 1.41 | 1.00 | 0  | 5.00 | 6.40 |
| **D (5,7)** | 7.21 | 5.83 | 5.00 | 0  | 1.41 |
| **E (6,8)** | 8.49 | 7.07 | 6.40 | 1.41 | 0  |

---

### **Step 2: Merge the Closest Pair (B, C → BC)**
- **Distance(B, C) = 1.00** (Smallest in the matrix)
- Merge **B and C** into **BC**.

Updated Clusters:
```
Level 1: {A, BC, D, E}
```

---

### **Step 3: Merge the Next Closest Pair (D, E → DE)**
- **Distance(D, E) = 1.41**
- Merge **D and E** into **DE**.

Updated Clusters:
```
Level 2: {A, BC, DE}
```

---

### **Step 4: Merge the Next Closest Pair (A, BC → ABC)**
- **Distance(A, BC) = 1.58**
- Merge **A and BC** into **ABC**.

Updated Clusters:
```
Level 3: {ABC, DE}
```

---

### **Step 5: Merge the Final Clusters (ABC, DE → ABCDE)**
- **Distance(ABC, DE) = 6.78**
- Merge **ABC and DE** into the final **ABCDE cluster**.

Updated Clusters:
```
Level 4: {ABCDE}
```

---

## **4. Correct Hierarchical Levels with Distance Similarity**
| **Level** | **Clusters at This Level** | **Merging Criteria** |
|-----------|----------------------------|------------------------|
| **Level 0** | {A, B, C, D, E} | Each data point is a separate cluster. |
| **Level 1** | {A, BC, D, E} | **B and C merge (d=1.00)**. |
| **Level 2** | {ABC, DE} | **D and E merge (d=1.41)**, **A merges with BC (d=1.58)**. |
| **Level 3** | {ABCDE} | **ABC and DE merge (d=6.78)** (final cluster). |

---

## **5. Determining Distance Similarity**
We now determine whether **BC and DE are at the same hierarchical level** by considering all relevant distances.

### **Key Distances:**
- \( d(B, C) = 1.00 \)
- \( d(D, E) = 1.41 \)
- \( d(A, BC) = 1.58 \)
- \( d(ABC, DE) = 6.78 \)

### **Step 1: Compute Relative Differences**
![image](https://github.com/user-attachments/assets/779e9991-b996-4fe2-ba0e-4408c09d4a6c)

### **Step 2: Compare with Threshold**
- **Distance(B, C) and Distance(D, E) are NOT very similar (29% difference).**
- **Distance(D, E) and Distance(A, BC) are similar (11% difference, within a 15% threshold).**

### **Final Similarity Conclusion**
- **BC merges first (B, C merge at distance = 1.00).**
- **DE and ABC are at the same level** because their merging distances (1.41 and 1.58) are similar.
- **The final merge (ABC, DE) happens at a much higher distance (6.78), so it is a separate level.**

---

## **6. Final Hierarchical Clustering Dendrogram**
```
          ABCDE        <-- Level 3 (Final Merge)
         /     \
       ABC     DE      <-- Level 2 (Merging at similar distances)
      /   \    /  \
     A    BC  D    E  <-- Level 1 (First merges)

```

✅ **Corrected Levels:**
1. **Level 0:** {A, B, C, D, E} (All individual clusters)
2. **Level 1:** {A, BC, D, E} (B, C merged)
3. **Level 2:** {ABC, DE} (A merges with BC; D merges with E)
4. **Level 3:** {ABCDE} (Final merge)

---

## **7. Key Takeaways**
✅ **Hierarchical clustering groups data step-by-step based on distance similarity.**  
✅ **Clusters merge at the same level when they merge at similar distances.**  
✅ **BC is a separate level from DE because their merging distances are different.**  
✅ **ABC and DE are at the same level because their distances (1.41 and 1.58) are similar.**  
✅ **The dendrogram visually represents the hierarchical clustering process.**  
