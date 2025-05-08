## 🔄 **Recurrent Connections & Hidden States**

### 🧠 **1. What Are Recurrent Connections?**

In a standard feedforward neural network, data flows in one direction—from input to output. However, **Recurrent Neural Networks (RNNs)** introduce a special kind of connection:
🔁 **Recurrent connections**, which loop back the output of a neuron to itself in the next time step.

This feedback loop allows RNNs to retain **temporal context**, making them uniquely suited for handling sequences like:

* 📖 Sentences in language models
* 🎵 Audio signals
* 📊 Time series data

💡 **Key idea**: The network can "remember" what it saw previously by feeding information forward through time.

---

### 📦 **2. Role of Hidden States**

The **hidden state** $h_t$ is the RNN’s **internal memory** at time step $t$. It captures both:

* 📥 The **current input** $x_t$
* 🧾 The **summary of past inputs** stored in $h_{t-1}$

It’s updated at every time step via:

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

✅ This means the hidden state evolves over time, integrating new inputs with past knowledge—enabling **context-aware processing**.

---

### 🔁 **3. Information Flow Over Time**

Visualize the RNN like this:

```
x₁ → h₁ → h₂ → h₃ → ... → hₜ
          ↑     ↑         ↑
          x₂    x₃        xₜ
```

* At each time step:

  * The hidden state gets updated using the current input and the previous hidden state.
  * This forms a **chain-like structure**, often called **"unrolling the RNN."**

📌 Because weights are shared across all time steps, the RNN is:

* 🧬 Lightweight (fewer parameters)
* 💡 Better at generalizing across sequence lengths

---

### 🧭 **4. Why Hidden States Matter**

Hidden states are the **core memory mechanism** of RNNs. They allow the network to:

* Track sentence structure in NLP 🗣️
* Understand rhythm in music generation 🎼
* Model dependencies in stock prices 📈

But:
⚠️ **Basic RNNs forget long-term dependencies** easily due to vanishing gradients. That's why LSTM and GRU were introduced.
