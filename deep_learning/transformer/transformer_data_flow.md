### 🔍 **Data Flow Through the Transformer**

![image](https://github.com/user-attachments/assets/3700dac7-fd6c-4a87-bbfb-2961731d903b)

---

## 🔁 **High-Level Overview**

### 🎯 Input:

A source sequence (e.g., a sentence in English):

> *“The cat sat.”*

### 🎯 Output:

A target sequence (e.g., translation in French):

> *“Le chat s'est assis.”*

---

## 🧭 **Step-by-Step Data Flow**

---

### 🔹 **1. Input Tokenization & Embedding**

* Text is broken into tokens (e.g., “The”, “cat”, “sat” → \[101, 402, 578])
* Each token is converted into a **dense vector (embedding)**.
* **Positional encoding** is added to inject word order.

$$
\text{Input} = \text{Embedding}(\text{Tokens}) + \text{Positional Encoding}
$$

---

### 🔹 **2. Encoder Stack**

Each token embedding is passed through **N identical encoder layers**.

For each encoder layer:

1. **Multi-head self-attention**:

   * Each token can **attend to all others** in the input.
   * Builds a **context-aware representation** of each word.

2. **Residual connection + LayerNorm**

3. **Feedforward network (FFN)**:

   * Applied independently to each token.

4. **Another residual + LayerNorm**

✅ **Output**: A sequence of **context-rich vectors**, one per input token.

---

### 🔹 **3. Decoder Input Prep**

* Decoder also receives tokenized input (the translation so far)
* During training, this is the target sentence shifted right
  (e.g., start with \[“<s>”, “Le”, “chat”] → predict “s'est”, then “assis”, etc.)
* Decoder input is also embedded + positional encoding added

---

### 🔹 **4. Decoder Stack**

Each decoder layer has **three sub-layers**:

1. **Masked Multi-head self-attention**

   * Prevents each token from seeing the **future** (auto-regression)
   * Example: At time step 3, only tokens 1–3 are visible

2. **Encoder–Decoder attention**

   * Decoder tokens **attend to encoder outputs**
   * Helps the model relate the translated words back to the source

3. **Feedforward network + residuals + normalization**

✅ Output: A sequence of vectors, one per decoder token

---

### 🔹 **5. Final Linear + Softmax**

* Decoder output is passed through a **linear projection layer**
* Then through **softmax** to get a probability distribution over the vocabulary
* The **highest probability token is chosen** (during inference)

---

## 📊 **Visual Flow Summary**

```
Input Sequence (tokens)
      ↓
Embeddings + Positional Encoding
      ↓
[ Encoder Layers (Self-Attn + FFN) ]
      ↓
Contextualized Encoder Outputs
      ↓                             ↑
Decoder Inputs (shifted targets)   |
      ↓                            |
Embeddings + Positional Encoding   |
      ↓                            |
[ Decoder Layers:                 ]
  - Masked Self-Attn              |
  - Encoder–Decoder Attention ←───
  - FFN                           ]
      ↓
Final Linear → Softmax → Output Tokens
```

---

### 🧠 One-Liner Summary:

> In a Transformer, data flows through layers of **attention and feedforward networks**, with encoders building rich context and decoders generating outputs step-by-step using that context.
