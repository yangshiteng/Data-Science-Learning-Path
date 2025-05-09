## 🚀 **Forward Propagation in RNNs**

### 🧭 **1. What Is Forward Propagation?**

In a Recurrent Neural Network (RNN), **forward propagation** refers to the process of passing inputs through the network over time to compute outputs.
Unlike feedforward networks that handle a single input at once, RNNs process **entire sequences**, updating their **hidden state** at each time step to reflect both current and past information.

---

### 🔄 **2. Step-by-Step Flow**

At each time step $t$, the network:

1. 📥 Receives input $x_t$
2. 🧠 Updates the hidden state $h_t$ based on the previous state $h_{t-1}$ and current input
3. 📤 Produces an output $y_t$

Mathematically:

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

$$
y_t = W_{hy}h_t + b_y
$$

🔍 **Where**:

* $W_{xh}$: Input-to-hidden weights
* $W_{hh}$: Hidden-to-hidden (recurrent) weights
* $W_{hy}$: Hidden-to-output weights
* $b_h, b_y$: Bias terms
* $\tanh$: Activation function introducing non-linearity

---

### 🧬 **3. Sequential Computation**

Forward propagation is performed **sequentially** through the input sequence $x_1, x_2, ..., x_T$.
This process is often visualized as **unrolling the RNN**:

```
Time Step:   t=1     t=2     t=3     ...     t=T
             ↓       ↓       ↓               ↓
Inputs:     x₁  →   x₂  →   x₃  →   ...   →  xₜ
             ↓       ↓       ↓               ↓
Hidden:     h₁  →   h₂  →   h₃  →   ...   →  hₜ
             ↓       ↓       ↓               ↓
Outputs:    y₁     y₂     y₃     ...      yₜ
```

📌 Each time step’s output depends not only on the **current input**, but also on the **entire history of previous inputs**, encoded in the hidden state.

---

### ⚠️ **4. Computational Characteristics**

* 🔁 **Weight sharing**: The same weights are used at every time step, making RNNs efficient and scalable to variable-length sequences.
* 🔄 **Memory through recurrence**: Past information is embedded in the hidden state, enabling temporal understanding.

---

### 🛠️ **5. Forward Pass in Practice**

To perform a full forward pass over a sequence:

1. Set $h_0 = 0$ (or learnable)
2. For $t = 1$ to $T$:

   * Compute $h_t$
   * Compute $y_t$ (optional at every step or only final step)

This results in a list of hidden states and possibly a list of outputs, depending on the task (e.g., sequence-to-sequence vs. sequence-to-one).
