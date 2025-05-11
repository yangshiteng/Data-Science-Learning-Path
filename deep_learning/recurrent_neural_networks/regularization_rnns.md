## 🛡️ **Regularization Techniques for RNNs**

Regularization helps prevent your RNN from **overfitting** — learning the training data too well, including its noise or quirks — by encouraging the model to generalize better to unseen data.

---

## 🔽 **1. Dropout in RNNs**

### ✅ What is Dropout?

**Dropout** randomly “drops out” (i.e., sets to zero) a subset of neurons during training, forcing the network to learn **redundant representations**.

### 🔧 How It Works:

During training, for each forward pass:

* With probability $p$, **zero out** a neuron's output.
* At test time, scale the outputs or use the full network.

$$
\text{Dropped Output} = h_t \cdot \text{Bernoulli}(1 - p)
$$

### 🧠 Why It’s Tricky in RNNs:

* In feedforward layers, dropout works well.
* But **naive dropout in recurrent connections** breaks temporal consistency (the same unit is randomly dropped at each time step, causing noise).

### ✅ Solution:

* Use **dropout on input/output layers only**, or
* Use **variational dropout**: apply the **same dropout mask** at every time step to preserve temporal consistency.

### 🔍 Where It’s Applied:

* **Input dropout**: on the input $x_t$
* **Output dropout**: on the hidden state $h_t$
* **Recurrent dropout**: between time steps (requires special handling)

---

## 🔁 **2. Zoneout**

### ✅ What is Zoneout?

**Zoneout** is a regularization method designed specifically for **RNNs**, and especially LSTMs.

Instead of zeroing out units like dropout, **Zoneout randomly "freezes" units**, meaning:

> At each time step, some hidden units are **kept the same as in the previous time step**.

$$
h_t = \text{mask} \cdot h_{t-1} + (1 - \text{mask}) \cdot \tilde{h}_t
$$

Where:

* $\tilde{h}_t$ is the new hidden state
* $h_{t-1}$ is the previous hidden state

### 🧠 Why Zoneout Works:

* Preserves the **temporal structure** of sequences better than dropout
* Encourages the model to **carry forward memory** more robustly
* Helps prevent overfitting in deep RNNs or long sequences

### 📌 Commonly used in:

* LSTM and GRU cells
* Tasks requiring long-term memory (e.g., language modeling)

---

## 🧾 Summary Table

| Technique   | Works by...                        | Good for...                 | Note                                |
| ----------- | ---------------------------------- | --------------------------- | ----------------------------------- |
| **Dropout** | Zeroing out units randomly         | Input/output regularization | Use same mask per time step in RNNs |
| **Zoneout** | Randomly keeping old hidden states | Preserving temporal memory  | More natural for RNNs than dropout  |

---

## 🧠 Final Tip

> For RNNs, **Dropout is still useful**, but **Zoneout** is more tailored to their sequential nature.
> For best results, use dropout on **input/output layers**, and **Zoneout or variational dropout** in the **recurrent layers**.
