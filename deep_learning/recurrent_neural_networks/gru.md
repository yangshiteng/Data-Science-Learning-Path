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

![image](https://github.com/user-attachments/assets/1582466a-6d9b-4333-bcf1-46578f8e9b05)

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

![image](https://github.com/user-attachments/assets/326c4279-c88f-4fc4-9304-eb115e677e8d)

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
