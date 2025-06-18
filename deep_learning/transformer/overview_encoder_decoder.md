### 🔍 **Overview of Encoder and Decoder in the Transformer Architecture**

---

### ✅ **Quick Summary:**

The Transformer architecture is made up of two main components:

| Component   | Purpose                       |
| ----------- | ----------------------------- |
| **Encoder** | Understand the input sequence |
| **Decoder** | Generate the output sequence  |

These two work **together**, especially in tasks like translation, where the model reads an entire sentence (encoder), then produces a translated version (decoder), one token at a time.

---

### 🧱 **High-Level Diagram:**

```
[Input Sequence] → [Encoder Stack] → [Context Vectors]
                                       ↓
                           [Decoder Stack] → [Output Sequence]
```

---

## 🧠 **The Encoder**

### 🔄 What it does:

* Takes in the **entire input sequence** (e.g., a sentence)
* Computes **context-rich embeddings** for each token using **self-attention**
* Outputs a set of vectors that represent the input with all its relationships considered

### 🧩 Encoder Structure:

Each encoder block contains:

1. **Multi-head self-attention**
2. **Add & Norm (residual connection + layer normalization)**
3. **Feedforward layer (position-wise MLP)**
4. **Add & Norm**

There are usually **N identical encoder layers** stacked on top of each other (e.g., 6 layers in the original paper).

---

## 🧠 **The Decoder**

### 🔄 What it does:

* Takes the **encoder’s output** + what’s been generated so far (partial output)
* Uses attention to generate the **next word/token**
* Continues generating until the full sequence is complete (e.g., end-of-sentence token)

### 🧩 Decoder Structure:

Each decoder block has **three main sub-layers**:

1. **Masked Multi-head self-attention**

   * Prevents cheating by blocking access to future tokens during training
2. **Encoder-Decoder attention**

   * Lets the decoder attend to the encoder's output (i.e., the input sequence)
3. **Feedforward layer**

   * Same as in the encoder

Also includes **residual connections + normalization**.

---

### 📊 Summary Table:

| Feature        | Encoder                        | Decoder                                           |
| -------------- | ------------------------------ | ------------------------------------------------- |
| Input          | Entire source sequence         | Previously generated output tokens                |
| Attention type | Self-attention only            | Masked self-attention + encoder-decoder attention |
| Output         | Context-aware token embeddings | Next token prediction probabilities               |

---

### 💬 Analogy:

> Think of the encoder as **reading and understanding** an article.
> The decoder is like **writing a summary**, one word at a time, using what it understood from the article.
