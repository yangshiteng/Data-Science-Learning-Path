### 🔍 **Key Components of the Transformer Block**

**Layer Normalization, Residual Connections, and Feedforward Layers**

---

In both **encoder** and **decoder** blocks, there are 3 key building blocks that ensure the model trains well and generalizes effectively:

---

## 🧱 1. Residual Connections (Skip Connections)

### ✅ **What it is:**

A **shortcut** that adds the input of a layer directly to its output before passing to the next step.

### 📐 **Formula:**

$$
\text{Output} = x + \text{SubLayer}(x)
$$

### 🔍 **Why it matters:**

* Helps gradients flow during backpropagation
* Prevents "vanishing gradients"
* Encourages **identity learning** — the model can choose to pass information unchanged if needed

> 📘 Think of this like saying: "Let’s do something new with the input, **but keep a copy of the original just in case**."

---

## 🧪 2. Layer Normalization (LayerNorm)

### ✅ **What it is:**

A normalization technique that **stabilizes** the outputs across each token's feature vector.

### 🔍 **Why it matters:**

* Speeds up training
* Makes the model **less sensitive** to weight initialization
* Works well with residuals

### 📐 **Where it's used:**

Typically applied **after the residual connection**:

$$
\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))
$$

> 🧠 Unlike batch normalization, which works across a batch, LayerNorm works **within each token vector**.

---

## 🔁 3. Feedforward Network (FFN or MLP)

### ✅ **What it is:**

A simple **two-layer fully connected neural network** applied **independently to each token**.

### 📐 **Formula:**

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

### 🔍 **Why it matters:**

* Adds **non-linearity** and **depth** beyond attention
* Helps the model learn **richer representations** for each token

> 📘 Even though attention captures relationships **between tokens**, the FFN helps process each token **individually** with more flexibility.

---

## 🧱 Summary: Transformer Block Structure

For each encoder or decoder layer:

```
1. SubLayer 1: Multi-head attention
   → Add + LayerNorm

2. SubLayer 2: Feedforward network
   → Add + LayerNorm
```

---

### 🧠 One-Liner Summary:

> **Residuals help information flow**, **LayerNorm keeps it stable**, and **Feedforward layers add learning depth** — all working together to make the Transformer powerful and trainable.
