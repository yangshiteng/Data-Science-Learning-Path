# 🏗️ **Hierarchical Recurrent Neural Networks (Hierarchical RNNs)**

---

## 📘 **What Is a Hierarchical RNN?**

A **Hierarchical RNN** is an architecture that models sequences **at multiple levels of abstraction or granularity** by stacking RNNs **across time scales or structural layers**.

### ✅ In simple terms:

* It **captures short-term dependencies** at a fine-grained level (e.g., words in a sentence)
* Then **models longer-term dependencies** at a coarse level (e.g., sentences in a document)

This is done using **multiple RNN layers**, where **each layer processes a different temporal unit** (e.g., words → sentences → paragraphs).

---

## 🧱 **Architecture Example: Document Modeling**

Consider a document composed of multiple sentences, and each sentence composed of words.

1. **Word-level RNN (lower layer)**
   Processes words in a sentence
   Outputs a **sentence representation**

2. **Sentence-level RNN (higher layer)**
   Processes sentence vectors
   Outputs a **document representation**

---

### 🔄 Forward Flow:

![image](https://github.com/user-attachments/assets/a0398e97-8da5-4e06-92e5-87991b7bcecb)

---

## 🧠 **Why Use Hierarchical RNNs?**

| Problem                          | How Hierarchical RNN Helps                                |
| -------------------------------- | --------------------------------------------------------- |
| Long sequences (e.g., documents) | Breaks them into chunks and models higher-level structure |
| Multi-scale patterns             | Captures local and global dependencies                    |
| Sparse long-term links           | Easier to model when grouped by structure                 |

---

## 🔍 **Use Cases**

* 📝 **Document classification**
* 🤖 **Dialogue modeling** (utterance-level and conversation-level)
* 🎥 **Video understanding** (frame-level and shot-level)
* 🧠 **Multimodal learning** (word-level text + sentence-level audio or image embeddings)

---

## ✅ **Advantages**

| Feature                     | Benefit                                           |
| --------------------------- | ------------------------------------------------- |
| 🧱 Structure-aware          | Learns language or sequence hierarchy             |
| 🧠 Better memory management | Each level focuses on manageable-length sequences |
| 📈 Higher accuracy          | Particularly in document-level NLP tasks          |

---

## ⚠️ **Limitations**

| Limitation                | Reason                                        |
| ------------------------- | --------------------------------------------- |
| 🧮 More computation       | Multiple RNNs at different levels             |
| 🧾 Preprocessing required | Must split sequences into structured subunits |
| 🛠️ Model complexity      | Harder to implement and tune than flat RNNs   |

---

## 🔧 PyTorch-style Pseudocode:

```python
# Word-level RNN
word_outputs, _ = word_rnn(word_inputs)

# Aggregate sentence vector (e.g., last hidden state)
sentence_vector = word_outputs[-1]

# Sentence-level RNN
document_outputs, _ = sent_rnn(sentence_vectors)
```

---

## 🧾 Summary

| Feature      | Description                                     |
| ------------ | ----------------------------------------------- |
| What it does | Models sequential data across multiple levels   |
| Best for     | Long, structured sequences (text, video, audio) |
| Architecture | Stacked RNNs across time-scales                 |
| Key benefit  | Captures both local and global dependencies     |
