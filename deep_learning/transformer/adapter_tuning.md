# 🔌 **Adapter Tuning in Transformers**

---

## 🧠 What Is Adapter Tuning?

**Adapter tuning** is a method for **efficiently fine-tuning large pretrained Transformer models** by inserting **small trainable layers (adapters)** between the frozen layers of the model.
Instead of updating all model parameters, **only the adapters are trained** for a given task.

> 📌 The rest of the model (e.g., BERT, T5, GPT) stays **frozen**.
> 🧩 Only the **adapters** learn task-specific knowledge.

---

## 🧩 Why Adapter Tuning?

Training huge Transformer models (like BERT with 110M or GPT-3 with 175B parameters) is:

* 🛑 **Resource-intensive**
* 🛑 **Slow**
* 🛑 **Hard to scale across tasks**

Adapter tuning solves this by:

* ✅ Reducing trainable parameters to **<5%** of the model
* ✅ Allowing **multi-task** or **domain adaptation** efficiently
* ✅ Preserving the **general knowledge** of the base model

---

## ⚙️ How Adapter Tuning Works – Step by Step

---

### 🔹 Step 1: Start with a Pretrained Model

Use a pretrained model such as:

* BERT, RoBERTa (Encoder-based)
* T5, BART (Encoder–Decoder)
* GPT (Decoder-based)

**Do not modify or unfreeze it**.

---

### 🔹 Step 2: Insert Adapter Modules into Each Layer

Adapters are small bottleneck layers added between layers of the Transformer:

```
[Input]
   ↓
LayerNorm
   ↓
→ [Adapter: Down → Nonlinear → Up] → Add & Norm →
   ↓
[Next Layer]
```

Each adapter typically has:

* **Down-projection**: reduces dimensionality (e.g., 768 → 64)
* **Activation**: e.g., ReLU or GELU
* **Up-projection**: restores dimension (64 → 768)

🔁 This bottleneck structure:

* Adds capacity for learning
* Keeps the parameter count small

---

### 🔹 Step 3: Train Only the Adapters

* Freeze the base model parameters (weights are not updated)
* **Only train the small adapter modules**
* Optionally train a **task head** (like a classifier)

---

## 🧠 Adapter Module Architecture

Let’s assume base hidden size $d = 768$, adapter bottleneck size $m = 64$:

1. Input: $h \in \mathbb{R}^{768}$
2. Down-projection: $W_{down} \in \mathbb{R}^{64 \times 768}$
3. Nonlinearity: ReLU or GELU
4. Up-projection: $W_{up} \in \mathbb{R}^{768 \times 64}$
5. Output: $h + W_{up}(f(W_{down}(h)))$

> 🔁 This residual connection ensures stability and compatibility.

---

## 📊 Why This Works

Transformers already have **rich pretrained knowledge**.
Adapters provide **just enough capacity** to specialize in a new task without overwriting general knowledge.

---

## 🛠️ Implementation Options

### 📦 Tools & Libraries:

* **AdapterHub**: Plug-and-play adapters for Hugging Face models
* **Hugging Face PEFT** (`peft.adapters`)
* **transformers.adapters** (`model.add_adapter(...)`)

### 🧪 Training Setup:

* Optimizer: AdamW
* LR: Usually higher than fine-tuning (since it's a small set)
* Training time: 10x faster than full fine-tuning
* Output: A few MB per adapter

---

## 🧪 Use Cases

| Scenario                         | Benefit                            |
| -------------------------------- | ---------------------------------- |
| ✅ Fine-tuning on limited compute | Lightweight tuning only            |
| ✅ Multi-task learning            | One model, many adapters           |
| ✅ Domain-specific NLP            | Custom adapters for each domain    |
| ✅ Privacy (edge deployment)      | Only send adapter, not whole model |
| ✅ Continual learning             | Add adapters instead of retraining |

---

## 📦 Example: Multi-task Adapter Tuning with BERT

| Task       | Adapter Module           |
| ---------- | ------------------------ |
| Sentiment  | `bert_sentiment_adapter` |
| NER        | `bert_ner_adapter`       |
| Medical QA | `bert_medqa_adapter`     |

All adapters share the **same frozen base model** → modular & efficient.

---

## 📉 Performance vs. Efficiency

| Method           | Accuracy  | Trainable Params | Speed        |
| ---------------- | --------- | ---------------- | ------------ |
| Full Fine-Tuning | ⭐⭐⭐⭐⭐     | 100%             | ❌ Slow       |
| Adapter Tuning   | ⭐⭐⭐⭐      | \~1–5%           | ✅ Fast       |
| Prompt Tuning    | ⭐⭐ to ⭐⭐⭐ | <1%              | ✅✅ Very Fast |

📌 Adapter tuning **approaches full fine-tuning** performance while using much less memory and training time.

---

## 📁 Storage and Reuse

* Fine-tuned model = 400MB+
* Adapter = 5–20MB
* You can keep hundreds of adapters and **swap** them on-demand without reloading the base model.

---

## 🧠 One-Liner Summary:

> **Adapter tuning** is a modular, parameter-efficient tuning strategy that **adds small learnable layers** to a frozen Transformer model — letting you train fast, switch tasks, and scale affordably without touching the base model.
