## 📘 What are Recurrent Neural Networks (RNNs)?

**Recurrent Neural Networks (RNNs)** are a type of neural network designed to handle **sequential data**. This includes any data where the order of the elements matters—like text, speech, time-series data, and video frames.

The key idea behind RNNs is to use the output from the previous step as an input to the current step. This gives the network a kind of **memory**, allowing it to retain context across time.

---

## 🔄 How RNNs Work: The Core Concept

### Traditional Neural Networks vs RNNs

* A **Feedforward Neural Network (FNN)** processes input independently—no memory of past inputs.
* An **RNN** has loops in its architecture, allowing it to pass information from one step to the next.

### RNN Cell Structure

At each time step `t`, the RNN receives:

* Input vector `xₜ`
* Previous hidden state `hₜ₋₁` (memory)

It outputs:

* New hidden state `hₜ`

**Mathematically:**

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

$$
y_t = W_{hy} h_t + b_y
$$

Where:

* `W_{hh}`, `W_{xh}`, `W_{hy}` are weight matrices
* `b_h`, `b_y` are bias terms
* `tanh` is an activation function introducing non-linearity

### Example RNNs Architecture

![image](https://github.com/user-attachments/assets/4806ccf0-d74b-4597-a597-45769dadaa88)

#### 🔄 **1. Folded View (Left Side)**

This shows a **single RNN cell** with loops representing **recurrence**. It receives:

* **Input $x$** (green circle)
* **Hidden state $h$** from the previous time step (blue box)
* And it produces:

  * A **new hidden state**
  * An **output $o$** (pink circle)

Arrows and Matrices:

* **U**: Weight matrix from input $x$ to hidden state $h$
* **V**: Weight matrix from hidden state $h$ to output $o$
* **W**: Weight matrix for the recurrent connection (from previous $h$ to current $h$)

This compact view summarizes the inner logic of the RNN.

#### 📆 **2. Unfolded View (Right Side)**

The large arrow shows how the single RNN cell is **unfolded over time**. Each RNN cell (blue box) represents the **same RNN unit with shared weights** at different time steps $t-1$, $t$, and $t+1$.

Step-by-Step Breakdown:

| Time Step | Input     | Hidden State | Output    |
| --------- | --------- | ------------ | --------- |
| $t-1$     | $x_{t-1}$ | $h_{t-1}$    | $o_{t-1}$ |
| $t$       | $x_t$     | $h_t$        | $o_t$     |
| $t+1$     | $x_{t+1}$ | $h_{t+1}$    | $o_{t+1}$ |

Each step:

1. Takes the current input $x_t$
2. Combines it with the previous hidden state $h_{t-1}$
3. Computes a new hidden state $h_t$
4. Produces an output $o_t$

---

## 🔁 Sequence Processing in RNNs

Suppose you’re processing the sentence “I love AI” word by word:

1. First word: input “I” → output and update memory
2. Second word: input “love” + memory from “I”
3. Third word: input “AI” + memory from “love”

This memory mechanism makes RNNs powerful for context-aware tasks.

---

## 🧠 Variants of RNNs

Due to the limitations of simple RNNs, several improved architectures were developed:

### 1. **LSTM (Long Short-Term Memory)**

* Introduced memory cells and gates (input, output, forget) to combat vanishing gradients.
* Better at learning **long-term dependencies**.

### 2. **GRU (Gated Recurrent Unit)**

* Similar to LSTM but with fewer gates (reset and update gates).
* Faster and computationally more efficient.

---

## ⚙️ Applications of RNNs

| Field            | Use Case                              |
| ---------------- | ------------------------------------- |
| Natural Language | Machine translation, text generation  |
| Speech           | Speech-to-text, voice recognition     |
| Time Series      | Stock prediction, weather forecasting |
| Music & Video    | Music generation, video captioning    |
| Healthcare       | Patient monitoring over time          |

---

## ✅ Advantages of RNNs

* Natural fit for **sequential and time-series data**
* Shared parameters across time = fewer parameters
* Can model **context and temporal dynamics**

---

## ❌ Limitations of RNNs

* **Vanishing and exploding gradients** during training
* Hard to capture **long-term dependencies**
* **Slow training** due to sequential data processing
* Less effective compared to newer models like **Transformers**

---

## 🔚 Summary

| Feature             | RNN                       |
| ------------------- | ------------------------- |
| Input type          | Sequence                  |
| Memory              | Yes (hidden state)        |
| Training challenges | Vanishing gradients       |
| Better variants     | LSTM, GRU                 |
| Common replacements | Transformers in NLP tasks |
