## 🧾 **Vanilla RNN Formulation**

### 🌱 **1. What Is a Vanilla RNN?**

A **Vanilla RNN** is the simplest form of a Recurrent Neural Network. It serves as the foundational architecture for more advanced variants like LSTM and GRU.
Despite its simplicity, it captures the core idea of RNNs: **learning from sequential data using a hidden state that carries information over time.**

![image](https://github.com/user-attachments/assets/32efad40-a831-4265-a71f-f674ec4e6a85)

---

### 🧠 **2. Core Equations**

At each time step $t$, the Vanilla RNN updates its **hidden state** and computes an **output** using these equations:

$$
\textcolor{green}{\textbf{Hidden state:}} \quad h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

$$
\textcolor{blue}{\textbf{Output:}} \quad y_t = W_{hy}h_t + b_y
$$

🔍 **Variables**:

* $x_t$: Input vector at time step $t$
* $h_t$: Hidden state (memory)
* $y_t$: Output
* $W_{xh}, W_{hh}, W_{hy}$: Weight matrices
* $b_h, b_y$: Bias vectors
* $\tanh$: Activation function (can also be ReLU or sigmoid)

---

### 🔁 **3. Recurrent Computation**

The core idea is that the **hidden state** $h_t$ captures information from:

* 🔙 **Past**: via $h_{t-1}$
* 📥 **Present**: via $x_t$

This gives the network a memory of prior inputs—ideal for modeling sequences like:

* 🧾 Sentences in NLP
* 🎶 Melodies in music
* 🔢 Time series data

---

### 🔗 **4. Weight Sharing**

A key property of Vanilla RNNs is **parameter sharing** across time steps:

* The same weights $W_{xh}, W_{hh}, W_{hy}$ are used at each time step.
* This keeps the model compact and efficient, regardless of sequence length.

🧠 Think of it as the same neuron being copied across time with memory passed along.

---

### ⚠️ **5. Limitations**

Despite its elegance, Vanilla RNNs have some serious issues:

* 🕳️ **Vanishing gradients**: Hard to learn long-term dependencies
* 💥 **Exploding gradients**: Training can become unstable
* 🧠 **Short-term memory**: Struggles with retaining distant context

These issues often make it necessary to switch to more robust variants like **LSTM** and **GRU** for real-world tasks.

---

### 🧮 **6. Use Case Example (Language Modeling)**

Given a sentence like:

`"The cat sat on the ___"`

A Vanilla RNN would process each word step-by-step, updating its hidden state, and finally predict the next word based on its accumulated memory.
