# 🧠 **Long Short-Term Memory (LSTM)**

---

## 🔍 **1. What is LSTM?**

**Long Short-Term Memory (LSTM)** is a specialized type of **Recurrent Neural Network (RNN)** designed to **learn long-term dependencies** in sequence data. It was introduced by **Hochreiter & Schmidhuber in 1997** to address the **vanishing gradient problem** that plagues standard (vanilla) RNNs.

---

### 📌 Why was LSTM invented?

Standard RNNs struggle to remember information over long sequences due to:

* ⚠️ **Vanishing gradients** – gradients shrink, causing earlier time steps to have little influence
* ⚠️ **Exploding gradients** – gradients grow rapidly, destabilizing training

LSTM solves this by introducing a **memory cell** and **gating mechanisms** that **control the flow of information** through time.

---

## 🧱 **2. Core Idea of LSTM**

At each time step $t$, an LSTM cell maintains:

* A **cell state** $C_t$ → long-term memory
* A **hidden state** $h_t$ → short-term output
* Three **gates** to control what to forget, remember, and output

These gates decide **what information to keep, discard, or update**.

---

## 🔐 **3. LSTM Cell Components**

### ✅ Inputs:

* $x_t$: Input at time step $t$
* $h_{t-1}$: Previous hidden state
* $C_{t-1}$: Previous cell state

### ✅ Outputs:

* $h_t$: Current hidden state
* $C_t$: Current cell state

---

### ⚙️ **Gates and Equations**

1. ### 🧹 **Forget Gate** $f_t$

Decides **what to forget** from the previous cell state.

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

> Outputs a value between 0 (forget everything) and 1 (keep everything).

---

2. ### 📥 **Input Gate** $i_t$

Decides **what new information** to store in the cell state.

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

---

3. ### 💾 **Candidate Cell State** $\tilde{C}_t$

Creates a vector of **new candidate values** to potentially add to the cell state.

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

---

4. ### 🧠 **Update Cell State** $C_t$

Combine old memory (scaled by forget gate) and new info (scaled by input gate):

$$
C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
$$

---

5. ### 📤 **Output Gate** $o_t$

Decides **what part of the cell state to output** as the new hidden state:

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

Then the final output (hidden state):

$$
h_t = o_t \cdot \tanh(C_t)
$$

---

### 🧮 **All Together**

An LSTM cell calculates:

$$
\begin{array}{rl}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \\
\tilde{C}_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) \\
C_t &= f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \\
h_t &= o_t \cdot \tanh(C_t)
\end{array}
$$

---

## 📈 **4. Advantages of LSTM**

| Feature            | Benefit                                    |
| ------------------ | ------------------------------------------ |
| Gating mechanisms  | Control memory flow across time            |
| Long-term memory   | Learns dependencies across 100+ time steps |
| Gradient stability | Reduces vanishing/exploding gradients      |
| Modularity         | Easy to stack for deeper RNNs              |

---

## 🚀 **5. LSTM in Practice**

### 🛠️ Common Applications:

* 📖 **Language Modeling**
* 🧠 **Text Generation**
* 🗣️ **Speech Recognition**
* 🎧 **Music Composition**
* 📉 **Time Series Forecasting**
* 🤖 **Chatbots and Dialogue Systems**

---

### 🧪 PyTorch Example:

```python
import torch.nn as nn

# Define a single-layer LSTM
lstm = nn.LSTM(input_size=100, hidden_size=128, num_layers=1)

# x: (seq_len, batch_size, input_size)
output, (hn, cn) = lstm(x)
```

---

## 🧾 Summary

| Component     | Role                                 |
| ------------- | ------------------------------------ |
| $f_t$         | Forget gate → what to discard        |
| $i_t$         | Input gate → what to add             |
| $\tilde{C}_t$ | Candidate memory → possible new info |
| $C_t$         | Cell state → long-term memory        |
| $o_t$         | Output gate → what to expose         |
| $h_t$         | Hidden state → output to next layer  |
