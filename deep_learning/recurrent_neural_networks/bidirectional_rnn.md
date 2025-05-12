# 🔁🔁 **Bidirectional RNNs**

---

## 📘 **What Is a Bidirectional RNN?**

A **Bidirectional RNN** is an extension of a standard RNN where two RNNs are run **in parallel**:

* One processes the sequence **forward** (left to right)
* The other processes the sequence **backward** (right to left)

At each time step, their outputs are **concatenated or combined**, giving the model access to **both past and future context** for each position in the sequence.

![image](https://github.com/user-attachments/assets/2b4e0e97-f729-4c9c-a79a-51ab750c3205)

---

## 🧠 **Why Use a Bidirectional RNN?**

In many sequence tasks, knowing **only the past** (as in regular RNNs) isn't enough.
Sometimes, understanding the meaning of a word depends on **what comes after it**.

### ✅ Examples:

* In **Named Entity Recognition**:

  > “He works at **Apple**.”
  > Seeing “works at” and “.” helps classify “Apple” correctly.

* In **Speech recognition** or **translation**, later words can clarify earlier ambiguous ones.

---

## 🧱 **Architecture Overview**

Let’s denote:

* $x_t$: input at time step $t$
* $\overrightarrow{h}_t$: hidden state from **forward** RNN
* $\overleftarrow{h}_t$: hidden state from **backward** RNN
* $h_t$: final hidden state at time $t$

$$
\overrightarrow{h}_t = \text{RNN}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1})
$$

$$
\overleftarrow{h}_t = \text{RNN}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})
$$

$$
h_t = \left[ \overrightarrow{h}_t ; \overleftarrow{h}_t \right]
$$

→ The final output $h_t$ is typically the **concatenation** of both directions.

---

## 🔁 **How It Works (Intuition)**

At every time step, the model “knows”:

* What has happened **so far**
* What will come **next**

This allows richer contextual understanding than a regular RNN.

---

## 🧰 **Where It's Used**

* 🧾 **Text classification** (context matters across whole sentence)
* 🧠 **Named Entity Recognition (NER)**
* 🗣️ **Speech recognition**
* 🧬 **DNA/RNA sequence modeling**
* 📜 **Document modeling**

---

## ✅ **Advantages**

| Feature                   | Benefit                                 |
| ------------------------- | --------------------------------------- |
| 🔁 Uses future context    | More accurate predictions per time step |
| 🧠 Richer representations | Combines both directions of information |
| 📈 Often better accuracy  | Especially in classification tasks      |

---

## ⚠️ **Limitations**

| Limitation                 | Why it matters                      |
| -------------------------- | ----------------------------------- |
| 🚫 Not usable in real-time | You must wait for the full sequence |
| 🐌 Higher computation      | Two RNNs instead of one             |
| 💾 Larger model size       | Doubles hidden state dimensions     |

---

## 🔧 Example (PyTorch)

```python
rnn = nn.LSTM(input_size=100, hidden_size=128, bidirectional=True)
```

* This creates a **bidirectional LSTM**
* Final hidden state at each step is size `2 × hidden_size` (forward + backward)

---

## 🧾 Summary

| Feature            | Bidirectional RNN                          |
| ------------------ | ------------------------------------------ |
| Structure          | Forward + backward RNNs                    |
| Output at time $t$ | Combined context from past and future      |
| Best for           | Context-aware NLP tasks                    |
| Drawback           | Not usable for streaming or real-time data |
