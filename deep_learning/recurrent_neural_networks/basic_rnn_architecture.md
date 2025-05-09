## 🎯 **Basic RNN Architecture (Vanilla RNN)**

![image](https://github.com/user-attachments/assets/dda805ad-b754-4369-8788-ff8dea98dc66)

![image](https://github.com/user-attachments/assets/445be076-1592-4f96-b215-1613d9d9c164)


### 🧠 **1. Overview**

The **Basic Recurrent Neural Network (RNN)** is designed to model **sequential data**. Unlike feedforward networks, it contains **loops** that allow information to persist over time—acting like a **memory**. This makes it ideal for tasks like:

* 📝 Text generation
* 🔊 Speech recognition
* 📈 Time series forecasting

It is also known as **Vanilla RNN** which is the simplest form of a Recurrent Neural Network. It serves as the foundational architecture for more advanced variants like LSTM and GRU.
Despite its simplicity, it captures the core idea of RNNs: **learning from sequential data using a hidden state that carries information over time.**

---

### 🔧 **2. Core Equations**

At each time step $t$, the RNN consists of:

* 📥 **Input vector $x_t$**: The current element in the input sequence
* 🔁 **Hidden state $h_t$**: Memory that carries information from previous time steps
* 📤 **Output $y_t$**: The output or prediction at time step $t$

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

### 📚 **4. Sequence Processing**

To process a sequence $x_1, x_2, ..., x_T$:

1. 🛑 Initialize $h_0 = 0$
2. 🔄 For each time step $t \in \{1, ..., T\}$:

   * Update hidden state $h_t$
   * Compute output $y_t$

This process is known as **unrolling the RNN over time**.

A key property is **parameter sharing** across time steps:

* The same weights $W_{xh}, W_{hh}, W_{hy}$ are used at each time step.
* This keeps the model compact and efficient, regardless of sequence length.

🧠 Think of it as the same neuron being copied across time with memory passed along.

---

### ⚠️ **5. Limitations**

Despite its simplicity, basic RNNs have some drawbacks:

* 🚫 **Short-Term Memory**: Struggles with long-range dependencies
* 📉 **Vanishing Gradients**: Gradients shrink as they backpropagate through time
* 📈 **Exploding Gradients**: Gradients can also grow too large, destabilizing training

🧪 These issues motivated the development of more advanced architectures like:

* 🔒 **LSTM (Long Short-Term Memory)**
* ⚙️ **GRU (Gated Recurrent Unit)**

### 🧮 **6. Use Case Example (Language Modeling)**

Given a sentence like:

`"The cat sat on the ___"`

A Vanilla RNN would process each word step-by-step, updating its hidden state, and finally predict the next word based on its accumulated memory.
