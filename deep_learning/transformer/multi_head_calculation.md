Let’s walk through a **simple, concrete example** of how **multiple attention heads** are calculated and combined.

To keep it beginner-friendly and easy to follow, we’ll use:

* Only **2 tokens** as input (e.g., “I” and “love”)
* **2 attention heads**
* Very small vector sizes (dimension = 2)
* **Fake numbers** for clarity — not trained weights

---

## 🧮 **Step-by-Step: Multi-Head Attention (Simplified)**

---

### 🔢 **Input**

Let’s say we have two word embeddings (after tokenization and embedding layer):

```plaintext
Token 1: “I”      → [1.0, 0.0]  
Token 2: “love”   → [0.0, 1.0]
```

Let’s call this the **input matrix**:

$$
X = \begin{bmatrix} 1.0 & 0.0 \\ 
0.0 & 1.0 \end{bmatrix}
$$

---

### 🧠 **Assume We Have 2 Attention Heads**

Each head has its own set of weight matrices for **Query (Q)**, **Key (K)**, and **Value (V)**.
Let’s define them manually for simplicity.

---

### ✅ **Head 1:**

```plaintext
W_Q1 = W_K1 = W_V1 = Identity matrix
        [[1, 0],
         [0, 1]]
```

Then for each token:

* Q1 = X @ W\_Q1 → same as X
* K1 = X @ W\_K1 → same as X
* V1 = X @ W\_V1 → same as X

Let’s compute dot product attention for “I”:

* Query: \[1, 0]
* Keys: \[1, 0] and \[0, 1]
* Dot products:

  * \[1, 0] · \[1, 0] = 1
  * \[1, 0] · \[0, 1] = 0

Apply softmax:

$$
\text{softmax}( [1, 0] ) ≈ [0.73, 0.27]
$$

Now compute the weighted sum over **values**:

$$
\text{Attention output}_\text{head1} = 0.73 \cdot [1, 0] + 0.27 \cdot [0, 1] = [0.73, 0.27]
$$

Do the same for “love”:

* Query: \[0, 1]
* Dot products: \[0, 1] · \[1, 0] = 0, \[0, 1] · \[0, 1] = 1
* softmax(\[0, 1]) → \[0.27, 0.73]
* Output: \[0.27, 0.73]

So, **head 1 output**:

$$
\text{Head1 Output} = \begin{bmatrix} 0.73 & 0.27 \\ 
0.27 & 0.73 \end{bmatrix}
$$

---

### ✅ **Head 2: Different Projections**

Let’s make W\_Q2 = W\_K2 = W\_V2 = \[\[0, 1], \[1, 0]] (flipping inputs)

Then:

* New Q2 = X @ W\_Q2 →

  * "I" → \[1, 0] @ \[\[0,1],\[1,0]] = \[0, 1]
  * "love" → \[0, 1] @ \[\[0,1],\[1,0]] = \[1, 0]

Now compute attention for “I”:

* Q = \[0, 1], K1 = \[0, 1], K2 = \[1, 0]
* Dot products:

  * \[0, 1] · \[0, 1] = 1
  * \[0, 1] · \[1, 0] = 0
* softmax(\[1, 0]) = \[0.73, 0.27]
* Values (same as keys here): \[0, 1] and \[1, 0]
* Output:
  0.73 \* \[0, 1] + 0.27 \* \[1, 0] = \[0.27, 0.73]

So, **head 2 output**:

$$
\text{Head2 Output} = \begin{bmatrix} 0.27 & 0.73 \\ 
0.73 & 0.27 \end{bmatrix}
$$

---

### 🔗 **Concatenate Head Outputs**

Now we stack them side-by-side:

$$
\text{Combined} = \begin{bmatrix}
\text{Head1}_{\text{“I”}} \| \text{Head2}_{\text{“I”}} \\
\text{Head1}_{\text{“love”}} \| \text{Head2}_{\text{“love”}} 
\end{bmatrix}
= 
\begin{bmatrix}
0.73 & 0.27 & 0.27 & 0.73 \\
0.27 & 0.73 & 0.73 & 0.27
\end{bmatrix}
$$

---

### 📤 **Final Linear Projection**

Let’s say the final projection matrix $W^O$ just averages the values (for simplicity):

$$
W^O = \frac{1}{2} \cdot \text{[[1, 0, 1, 0], [0, 1, 0, 1]]}
$$

Apply it:

$$
\text{Final Output} = \text{Combined} @ W^O
=
\begin{bmatrix}
(0.73+0.27)/2 & (0.27+0.73)/2 \\
(0.27+0.73)/2 & (0.73+0.27)/2 
\end{bmatrix}
=
\begin{bmatrix}
0.50 & 0.50 \\
0.50 & 0.50
\end{bmatrix}
$$

---

### 🎯 **Final Output Explanation**

Each token ends up with a **final embedding of \[0.5, 0.5]**, showing that:

* It blended the outputs from both heads
* Each head learned to focus differently
* The final projection combined them meaningfully
