### 🔍 **Part 8.1: Encoder-Only Transformers – Introduction to BERT**

---

### ✅ **Quick Summary:**

**BERT** (Bidirectional Encoder Representations from Transformers) is a **Transformer-based model** that uses **only the encoder** stack.
It’s designed to deeply understand language by looking at **all words in context**, both to the left and right — hence “bidirectional.”

---

## 🧠 **What is BERT?**

* Developed by Google in 2018
* One of the first models to **pre-train deep contextual representations** using Transformers
* Pre-trained on a large corpus (Wikipedia + BooksCorpus)
* Powers many NLP applications: search, question answering, chatbots, etc.

---

## 🧱 **BERT Architecture (Encoder-Only)**

| Component | Description                                                          |
| --------- | -------------------------------------------------------------------- |
| Input     | Tokenized sentence(s) + segment IDs                                  |
| Stack     | Multiple layers of Transformer **encoders** (e.g., 12 for BERT-base) |
| Output    | One embedding per input token                                        |

📌 No decoder → BERT does **not generate** sequences — it **understands** them.

---

## 🔁 **Training Objectives (How BERT Learns)**

### 1. **Masked Language Modeling (MLM)**

* Randomly masks 15% of input tokens during training
* Model learns to **predict the masked words**
* Example:

  > Input: *“The \[MASK] sat on the mat.”*
  > Predict: *“cat”*

This forces BERT to understand the **full context** of the sentence.

### 2. **Next Sentence Prediction (NSP)**

* Given two sentences, BERT predicts whether the second sentence **follows the first** in the original text.
* Helps BERT understand sentence-to-sentence relationships

---

## 💼 **What BERT is Good At**

| Task                    | Example                            |
| ----------------------- | ---------------------------------- |
| Sentence classification | Spam detection, sentiment analysis |
| Token classification    | Named Entity Recognition (NER)     |
| Question answering      | SQuAD                              |
| Sentence similarity     | Semantic search                    |
| Text embedding          | Universal sentence representations |

---

## 🔄 **How BERT Is Used**

BERT is typically **fine-tuned** on downstream tasks:

1. Add a small task-specific head (e.g., a classifier)
2. Train on your labeled data (e.g., IMDB sentiment dataset)
3. Use BERT’s encoder layers to provide powerful features

---

### 🧠 One-Liner Summary:

> **BERT** is an **encoder-only Transformer** trained to understand text deeply using **masked language modeling** — ideal for classification, QA, and language understanding tasks.
