# 🧩 **Prefix Tuning in Transformers**

---

## 📘 What Is Prefix Tuning?

**Prefix Tuning** is a **parameter-efficient** method for adapting large pretrained Transformer models to downstream tasks **without modifying the original model weights**.

Instead of:

* Fine-tuning all model parameters (very expensive), or
* Using input-level prompt tokens (as in prompt tuning),

**Prefix Tuning trains a small set of learnable vectors** (called a *prefix*) that are **prepended to the key and value matrices** inside the attention mechanism of every Transformer layer.

> **Goal**: Steer the model toward a task-specific behavior **without touching the backbone model**.

---

## 🧠 Why Use Prefix Tuning?

| Challenge                               | Prefix Tuning Solution                         |
| --------------------------------------- | ---------------------------------------------- |
| Fine-tuning is resource-heavy           | Only tune a small number of parameters         |
| Risk of forgetting pretrained knowledge | Model remains frozen (zero forgetting)         |
| Need for modularity across tasks        | Different prefix for each task = plug and play |
| Want high control over generation       | Prefixes affect attention at every layer       |

---

## 🧠 Intuition

### In regular Transformers:

```plaintext
Input → Token Embedding → Transformer Layers → Output
```

Each layer uses:

$$
\text{Attention}(Q, K, V)
$$

Where:

* Q = Query from current token
* K, V = Keys and Values from all tokens in sequence

---

### In Prefix Tuning:

You prepend **learned key-value vectors** to each layer:

$$
\text{Attention}(Q, [K_{\text{prefix}}, K], [V_{\text{prefix}}, V])
$$

* These **prefix vectors are trainable** and shared across inputs.
* You don’t change the input, only **modify internal computations**.

---

## ⚙️ How Prefix Tuning Works (Step-by-Step)

---

### 🔹 Step 1: Freeze the Base Model

Use a pretrained model such as:

* GPT-2 / GPT-J / LLaMA (decoder-only)
* T5 / BART (encoder-decoder)

**All weights are frozen** — you won't update them.

---

### 🔹 Step 2: Add Learnable Prefix Embeddings

Define a set of **prefix vectors**:

* Size: $L_p$ tokens per layer
* Shape: $[L_p, d_{\text{model}}]$

Each prefix is mapped to:

* A **key prefix**: $[L_p, d_k]$
* A **value prefix**: $[L_p, d_v]$

These are added to the key/value matrices in each attention layer.

---

### 🔹 Step 3: Training

You only:

* Feed regular inputs (no changes to actual tokens)
* Update the prefix key/value embeddings (can use MLP to generate them)
* Use standard loss functions (e.g., cross-entropy for generation)

---

## 🧠 Technical Architecture

### 🔧 Transformer Attention

In a Transformer block:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

With **prefix tuning**, modify:

$$
K' = [K_{\text{prefix}}; K]  
\quad  
V' = [V_{\text{prefix}}; V]
$$

So the updated attention is:

$$
\text{Attention}(Q, [K_{\text{prefix}}, K], [V_{\text{prefix}}, V])
$$

You can **generate the prefix vectors directly** or with an MLP from a shared prefix embedding.

---

## 📊 Advantages of Prefix Tuning

| Advantage                   | Explanation                                     |
| --------------------------- | ----------------------------------------------- |
| ✅ Parameter-efficient       | Only train a few million parameters (often <1%) |
| ✅ Model remains frozen      | Safe, prevents catastrophic forgetting          |
| ✅ Plug-and-play             | One base model, many prefix modules             |
| ✅ Good for generation tasks | Affects attention at each layer deeply          |
| ✅ Compatible with LLMs      | Works with GPT-2, GPT-3, T5, BART, etc.         |

---

## 📉 Comparison to Other Tuning Methods

| Method            | Trains What?                   | Model Frozen? | Good For                         |
| ----------------- | ------------------------------ | ------------- | -------------------------------- |
| Fine-tuning       | Entire model                   | ❌ No          | Highest accuracy, costly         |
| Prompt Tuning     | Embeddings at input only       | ✅ Yes         | Small tasks, few layers affected |
| Adapter Tuning    | Adapter modules between layers | ✅ Yes         | Modular multi-task               |
| **Prefix Tuning** | Prefix K/V vectors per layer   | ✅ Yes         | Strong control over attention    |

---

## 🧪 Example Use Cases

| Use Case            | Why Prefix Tuning Works Well                            |
| ------------------- | ------------------------------------------------------- |
| Summarization       | Prefix guides decoder to focus on summary-relevant info |
| Style transfer      | Prefix controls tone, formality, etc.                   |
| Dialogue generation | Prefix injects persona or domain knowledge              |
| Question answering  | Helps model focus on relevant span-like behavior        |
| Translation tuning  | Prefix encodes domain-specific language pairs           |

---

## 📦 Implementation Details

### 🔧 Typical Config:

* Prefix length: $L_p = 5 \sim 20$
* Projection: optional MLP (prefix embedding → K, V space)
* Layers: often **applied to every attention layer**
* Optimizer: Adam or AdamW

---

## 💾 Storage and Deployment

* Store prefix vectors or small prefix networks
* Usually <10MB per task (vs 1.5GB for a full fine-tuned GPT-2)
* Load at runtime into frozen backbone model

---

## 🧪 Experimental Results (From Paper)

In **"Prefix-Tuning: Optimizing Continuous Prompts for Generation" (Li & Liang, 2021)**:

* Tasks: table-to-text, summarization, translation
* Models: GPT-2, BART
* Prefix Tuning achieved **\~95–98%** of full fine-tuning performance with **<1%** of parameters trained

---

## 🛠 Tools & Libraries

* 🤗 Hugging Face **PEFT**: `PrefixTuningConfig`, `PeftModel`
* Microsoft **Prompt Toolkit**
* [Original GitHub repo (Li & Liang)](https://github.com/XiangLi1999/PrefixTuning)

---

## 📈 Performance Table

| Task                | Model | Full FT Accuracy | Prefix Accuracy | Params Tuned |
| ------------------- | ----- | ---------------- | --------------- | ------------ |
| Table-to-Text       | GPT-2 | 16.2 BLEU        | 15.8 BLEU       | 0.1%         |
| CNNDM Summarization | BART  | 44.2 ROUGE-L     | 43.6 ROUGE-L    | 0.2%         |

---

## 🧠 Summary

> **Prefix Tuning** adds a **small trainable prefix of key/value vectors** to each Transformer attention layer. It’s a highly efficient, effective way to adapt LLMs for downstream tasks — especially in **generation, style control, and modular learning** — without touching the pretrained model.

---

## ✅ Final Notes

* ✅ If you’re doing **text generation or translation**, prefix tuning is often better than prompt tuning.
* ✅ It scales to **very large models** (GPT-3, T5-11B).
* ✅ You can **swap prefixes** per task like plug-ins.
