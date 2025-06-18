### 🔍 **How Attention Scores Are Computed**

---

### ✅ **Goal:**

Understand how a model decides **which words to focus on** using **queries and keys** — the **"attention scores"** tell it how relevant each word is to the current word being processed.

---

### 🧠 **The Core Idea: Compare Query with Keys**

Every word creates:

* A **Query vector** (Q) → “What am I looking for?”
* A **Key vector** (K) for every other word → “What do you offer?”

To measure **relevance**, the model computes the **dot product** between the query and each key:

$$
\text{score} = Q \cdot K
$$

The higher the dot product, the more similar (or relevant) the key is to the query.

---

### 🔢 **Step-by-Step Breakdown:**

Let’s say we have 3 words in a sentence:

> *“The cat sat.”*

For each word:

* Compute a **Query vector** (Q)
* Compute **Key vectors** (K)
* Compute **Value vectors** (V)

Suppose you're calculating attention for **“sat”**:

1. Take its **Query** vector $Q_{\text{sat}}$

2. Compute dot products with:

   * $K_{\text{the}}$
   * $K_{\text{cat}}$
   * $K_{\text{sat}}$ (itself)

3. These become **raw attention scores**:

$$
\text{scores} = [Q \cdot K_{\text{the}},\ Q \cdot K_{\text{cat}},\ Q \cdot K_{\text{sat}}]
$$

---

### 🧮 **Scaling the Scores**

Large dot products can lead to very small gradients (softmax gets too confident). So we **scale** the scores by:

$$
\frac{Q \cdot K}{\sqrt{d_k}}
$$

Where $d_k$ is the dimension of the key vectors.

This stabilizes training.

---

### 🔁 **Softmax to Turn Scores into Weights**

Now apply **softmax** to the scaled scores:

$$
\text{Attention Weights} = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)
$$

This converts raw scores into **normalized weights** that add up to 1.
These weights tell the model **how much attention to pay** to each word.

---

### 📊 **Example (Simplified)**

| Word  | Dot Product Score | After Scaling | Attention Weight |
| ----- | ----------------- | ------------- | ---------------- |
| "the" | 2.0               | 1.0           | 0.10             |
| "cat" | 4.0               | 2.0           | 0.30             |
| "sat" | 6.0               | 3.0           | 0.60             |

So for **“sat”**, it mostly focuses on **itself**, and a bit on “cat”.

---

### 🧠 One-Liner Summary:

> Attention scores are computed by comparing **queries** to **keys** with dot products, scaling the results, and turning them into **weights** using softmax.
