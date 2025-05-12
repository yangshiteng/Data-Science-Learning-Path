## 🔬 **Other Variants of RNN Cells**

---

### 1. 🧠 **Minimal Gated Unit (MGU)**

* **Simplified version of GRU**
* Uses **only one gate** (update gate)
* Reduces computational load further than GRU

![image](https://github.com/user-attachments/assets/06f33a3d-80db-440f-8172-eb63800440ca)

✅ *Use when memory is limited or for very large datasets.*

---

### 2. 🔄 **Independently Recurrent Neural Networks (IndRNN)**

* Each neuron has **its own recurrence**, i.e., the hidden state update is **element-wise**, not matrix-based.
* Enables deeper RNNs and better gradient flow.

#### Key idea:

$$
h_t^{(i)} = \sigma(W x_t + u^{(i)} h_{t-1}^{(i)})
$$

✅ *Supports deep RNN stacks and long sequences.*

---

### 3. 📦 **Unitary/Orthogonal RNNs**

* Use **unitary or orthogonal matrices** for recurrent weights.
* Designed to **preserve gradient norm** over long time steps (solving vanishing/exploding gradient problems).

Examples:

* uRNN (Unitary RNN)
* EURNN (Efficient Unitary RNN)

✅ *Useful in audio, music, and physics modeling where long-term stability matters.*

---

### 4. ⚡ **Skip RNN / Clockwork RNN**

* Introduce **temporal sparsity** by **updating only parts** of the hidden state at a time or at **scheduled intervals**.
* More efficient for long sequences with **redundant frames** (e.g., video, time series).

✅ *Improves speed and efficiency when not all time steps are equally important.*

---

### 5. 🧬 **Quasi-Recurrent Neural Networks (QRNN)**

* Combines the **parallelism of CNNs** with **temporal modeling of RNNs**.
* Uses 1D convolutions + pooling instead of full recurrence.

✅ *Much faster than LSTM/GRU, ideal for GPUs.*

---

### 6. 🧠 **JANET (Just Another NET)**

* A minimal LSTM variant with **only the forget gate**.
* Still retains memory gating, but simplified for faster convergence.

---

### 7. 🌀 **Delta-RNN / Fast Weights RNNs**

* Introduce **second-order updates** or **fast weights** that learn rapidly changing short-term associations.
* Inspired by cognitive theories of working memory.

✅ *Used in low-latency systems and memory-constrained tasks.*

---

## 🧾 Summary Table

| RNN Variant     | Key Feature                     | Compared To         | When to Use                       |
| --------------- | ------------------------------- | ------------------- | --------------------------------- |
| **MGU**         | Single-gate GRU                 | GRU (simplified)    | Fast, low-resource tasks          |
| **IndRNN**      | Independent recurrence per unit | Deep RNNs           | Long sequences, deep networks     |
| **Unitary RNN** | Norm-preserving recurrence      | Vanilla RNN         | Long-term gradient preservation   |
| **QRNN**        | Convolutions + pooling          | LSTM                | High-speed sequence modeling      |
| **Skip RNN**    | Skips updates dynamically       | Any RNN             | Sparse signals or redundant input |
| **JANET**       | Forget gate only                | LSTM (simplified)   | Efficient memory modeling         |
| **Delta-RNN**   | Fast-weight memory dynamics     | RNN w/ memory model | Cognitive/neuro-inspired models   |
