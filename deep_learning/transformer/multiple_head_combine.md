### 🔍 **How Multiple Attention Heads Are Combined**

*(The final step in Multi-Head Attention)*

---

### ✅ **Goal:**

Understand how the outputs of multiple attention heads are **merged into a single representation** that the model can use in the next layer.

---

### 🧠 **Step-by-Step Summary**

Let’s say we have **h attention heads** (e.g., 8 or 12).
Each head independently performs:

$$
\text{head}_i = \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)
$$

This results in **h different output vectors**, one from each head.

---

### 🔄 **Step 1: Concatenate Outputs**

All the head outputs are **concatenated** (joined end-to-end) into one long vector:

$$
\text{MultiHead Output} = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)
$$

If each head outputs a vector of size $d_k$, the total size becomes:

$$
\text{Total dimension} = h \times d_k
$$

---

### 🔧 **Step 2: Final Linear Projection**

The concatenated vector is then passed through a **final linear layer** (a learned weight matrix $W^O$) to mix and transform the information:

$$
\text{Final Output} = \text{Concat}(...)\ W^O
$$

This transforms the combined features back to the desired model dimension (usually the same size as the input embedding, e.g., 512 or 768).

---

### 💡 **Why This Works**

* Each head contributes **unique information**.
* The final linear layer lets the model **combine and redistribute** that information intelligently.
* It ensures the output is compatible with the next Transformer layer.

---

### 🎯 **Visual Analogy**

> Imagine each head as a specialist giving a report. You stack all the reports (concatenation), then a final editor reads and rewrites the full document (projection) into a clean summary.

---

### 🧠 One-Liner Summary:

> Multiple attention heads are **concatenated**, then passed through a final linear layer to produce a unified, rich representation for the next layer.
