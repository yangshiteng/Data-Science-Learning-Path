# 🔁 **Gated Recurrent Units (GRUs)**

---

## 📘 **What is a GRU?**

**Gated Recurrent Unit (GRU)** is a type of recurrent neural network (RNN) architecture introduced by Cho et al. (2014) as a simpler alternative to LSTM.

Like LSTMs, GRUs are designed to **capture long-term dependencies** and mitigate the **vanishing gradient problem** — but with **fewer gates and parameters**, making them faster and easier to train.

---

## 🧠 **Why Use GRUs?**

Compared to LSTMs:

* GRUs **merge the forget and input gates into a single “update gate”**
* GRUs **combine the hidden and cell states into one**
* This makes GRUs **simpler, faster**, and often just as effective on many tasks

---

## 🧱 **GRU Architecture Overview**

Each GRU cell maintains a **hidden state** $h_t$, and updates it using:

* An **update gate** $z_t$
* A **reset gate** $r_t$
* A **candidate hidden state** $\tilde{h}_t$

---

## ✍️ **GRU Equations**

```markdown
$$
z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)
$$

$$
r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)
$$

$$
\tilde{h}_t = \tanh(W_h [r_t \cdot h_{t-1}, x_t] + b_h)
$$

$$
h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t
$$
```

---

## 🧮 **Explanation of Each Term**

| Symbol        | Meaning                                        |
| ------------- | ---------------------------------------------- |
| $z_t$         | **Update gate**: how much of the past to keep  |
| $r_t$         | **Reset gate**: how much of the past to forget |
| $\tilde{h}_t$ | **Candidate hidden state**: new memory         |
| $h_t$         | **New hidden state**: final blended state      |
| $\sigma$      | Sigmoid activation                             |
| $\tanh$       | Tanh activation                                |

---

## ✅ **Advantages of GRUs**

| Feature                 | Benefit                           |
| ----------------------- | --------------------------------- |
| Simpler structure       | Fewer gates than LSTM             |
| Fewer parameters        | Faster to train                   |
| Competitive performance | Comparable to LSTMs on many tasks |

---

## 🔧 **GRUs in Practice**

### 🔍 Used for:

* Language modeling
* Machine translation
* Time-series forecasting
* Speech and audio processing

---

## 🧾 **Summary**

| Component           | GRU Equivalent                    |
| ------------------- | --------------------------------- |
| Forget + Input gate | Merged into **Update gate** $z_t$ |
| No cell state       | Only a hidden state $h_t$         |
| Simpler math        | Just 3 main equations             |
| Faster training     | Yes                               |
