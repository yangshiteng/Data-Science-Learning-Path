# 🎯 **Attention-Enhanced RNNs**

---

## 📘 **What Is Attention in RNNs?**

**Attention-enhanced RNNs** (commonly referred to as **RNNs with Attention**) are an extension of the standard **Encoder–Decoder (Seq2Seq)** architecture.

### 🧠 Core Idea:

Instead of forcing the decoder to rely on a **single fixed-length vector** (the encoder’s final hidden state), attention allows the decoder to **dynamically focus on different parts of the input sequence** at each decoding step.

> It’s like giving the model the ability to “look back” at relevant words when generating each output word.

---

## 🔍 **Why Attention Helps**

### 🔴 Problem in vanilla Seq2Seq:

* The encoder compresses the **entire input sequence into one vector**
* This becomes a **bottleneck**, especially for long or complex sequences

### 🟢 Attention solution:

* The decoder gets **direct access to all encoder hidden states**
* It computes **weights** (attention scores) over the input sequence
* It forms a **context vector** by taking a **weighted sum** of the encoder states

---

## 🧱 **Architecture Overview**

![image](https://github.com/user-attachments/assets/087a28e9-bd90-47a0-9d43-8b3b3a00a502)

Let:

* $h_1, h_2, ..., h_T$: hidden states from the encoder (one per input token)
* $s_t$: decoder hidden state at time $t$

### 🔹 1. **Alignment Scores**

Calculate similarity between each encoder hidden state $h_i$ and the decoder state $s_t$:

$$
e_{t,i} = \text{score}(s_t, h_i)
$$

Common scoring methods:

* Dot product: $e_{t,i} = s_t^\top h_i$
* MLP: $e_{t,i} = v^\top \tanh(W[s_t; h_i])$

---

### 🔹 2. **Attention Weights**

Convert scores to probabilities:

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
$$

→ These are the **attention weights** for each input token.

---

### 🔹 3. **Context Vector**

Compute the context vector $c_t$ as the weighted sum of encoder hidden states:

$$
c_t = \sum_{i=1}^{T} \alpha_{t,i} \cdot h_i
$$

---

### 🔹 4. **Enhanced Decoder Step**

Use $c_t$ along with the decoder’s previous hidden state to predict the next output:

$$
\hat{y}_t = \text{Softmax}(W[s_t; c_t] + b)
$$

---

## 🎯 **What Attention Does (Visually)**

At each output step, the model "looks" at the most relevant parts of the input — just like how a human might refer back to earlier words in a sentence when translating.

---

## 🧠 **Benefits of Attention**

| Feature                      | Benefit                                              |
| ---------------------------- | ---------------------------------------------------- |
| 🔍 Dynamic focus             | Decoder chooses where to look in input               |
| 📈 Improved performance      | Especially for long and complex sequences            |
| 🧠 Interpretability          | You can **visualize what the model is attending to** |
| 🧾 No information bottleneck | Avoids the limitations of fixed-length context       |

---

## 🧰 **Common Applications**

* 🗣️ Machine Translation
* 📄 Text Summarization
* 🤖 Dialogue Systems
* 🎙️ Speech Recognition
* 📷 Image Captioning (with visual attention)

---

## 🚀 Extensions of Attention

* **Bahdanau Attention** (Additive attention)
* **Luong Attention** (Multiplicative/dot-product attention)
* **Self-Attention** (used in Transformers — each word attends to every other word in the same sequence)

---

## 🧾 Summary

| Component         | Role                                    |
| ----------------- | --------------------------------------- |
| Attention Weights | Decide which encoder states to focus on |
| Context Vector    | Summarized input for current step       |
| Decoder           | Uses context + hidden state to generate |
| Benefit           | More flexible and powerful decoding     |
