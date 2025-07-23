### ✅ **Quick Summary:**

To train a Transformer (like for translation, summarization, or language modeling), you need to prepare input-output pairs in a specific way — especially for the **encoder and decoder** to work together properly.

---

## 🔁 **High-Level Structure**

For **encoder-decoder models** (like the original Transformer, T5, or BART):

| Component   | Input During Training                       |
| ----------- | ------------------------------------------- |
| **Encoder** | Full input sequence (e.g., source sentence) |
| **Decoder** | Target output sequence, **shifted right**   |
| **Output**  | The next token in the target sequence       |

---

### 📘 **Example: Machine Translation**

Let’s say we’re translating:

* **Input (source)**:

  > *“The cat sat.”*

* **Target (expected output)**:

  > *“Le chat s'est assis.”*

---

### 🧩 What the Model Sees During Training

| Stage             | Input Provided             | Expected Output                    |
| ----------------- | -------------------------- | ---------------------------------- |
| **Encoder**       | `[The, cat, sat]`          | —                                  |
| **Decoder Input** | `[<s>, Le, chat, s', est]` | `[Le, chat, s', est, assis, </s>]` |

* `<s>` = start-of-sequence token
* Decoder input is **shifted right** by 1 token
* Output includes the **correct next token** at each step

This allows the decoder to **learn to predict the next token**, using:

* Its current input
* Encoder context (via encoder–decoder attention)
* Masking to **prevent looking ahead**

---

## 🧠 Key Concepts in the Training Setup

### 🔹 **Teacher Forcing**

* The decoder is fed the **ground truth** output from the training data (not its own past predictions).
* Helps speed up training and improve stability.

### 🔹 **Token Shifting**

* During training, the model **never sees the token it's predicting**.
* It learns to generate token `t+1` given tokens `0` through `t`.

### 🔹 **Padding and Masking**

* Sequences are padded to the same length for batching.
* Padding is **masked out**, so the model doesn’t attend to padding tokens.

---

### 🧠 One-Liner Summary:

> During training, the encoder sees the full input, and the decoder learns to **predict the next token** step-by-step, using **shifted outputs and attention to the encoder's context**.
