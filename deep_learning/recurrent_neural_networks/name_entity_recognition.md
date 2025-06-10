## 🧠 What is Named Entity Recognition (NER)?

![image](https://github.com/user-attachments/assets/8877286d-de1b-42d3-9b24-39f38fc90fa4)

**Named Entity Recognition (NER)** is a Natural Language Processing (NLP) task where the goal is to **identify and classify specific entities** in text into predefined categories, such as:

| Entity Type | Examples                |
| ----------- | ----------------------- |
| **PERSON**  | Barack Obama, Elon Musk |
| **ORG**     | Google, United Nations  |
| **LOC**     | New York, Himalayas     |
| **DATE**    | July 4th, 2025          |
| **MISC**    | Nobel Prize, COVID-19   |

---

### 🎯 Objective

Given an input sentence:

> "**Barack Obama** was born in **Hawaii** in **1961**."

NER should output:

* “Barack Obama” → **PERSON**
* “Hawaii” → **LOC**
* “1961” → **DATE**

---

## 🤖 Why Use RNNs for NER?

NER is a **sequence labeling** task — we assign a label to each word in a sentence.

* RNNs (especially **BiLSTM**) excel at **processing sequences**, capturing **contextual dependencies** in both directions.
* This is important because the meaning of a word depends on its **surrounding context**.

✅ Example:

* The word “Apple” could be a **fruit** or a **company**, depending on context:

  * “I ate an Apple.” → 🍎
  * “Apple released a new iPhone.” → 🏢

---

## 📄 Input & Output Structure

### Input Sentence:

> “Steve Jobs founded Apple in 1976.”

### Tokenized:

`["Steve", "Jobs", "founded", "Apple", "in", "1976", "."]`

### Label Format (BIO scheme):

| Token   | Label  |
| ------- | ------ |
| Steve   | B-PER  |
| Jobs    | I-PER  |
| founded | O      |
| Apple   | B-ORG  |
| in      | O      |
| 1976    | B-DATE |
| .       | O      |

* **B-XXX**: Beginning of entity type XXX
* **I-XXX**: Inside entity
* **O**: Outside of any entity

---

## 🏗 RNN-Based NER Architecture

### Typical Pipeline:

```
Tokenized Text → Embedding Layer → BiLSTM → Dense Layer → Softmax (per token)
```

### Breakdown:

1. **Embedding Layer**:

   * Converts words into dense vectors (e.g., Word2Vec, GloVe, or contextual embeddings).
   * Shape: (sequence\_length, embedding\_dim)

2. **BiLSTM Layer**:

   * Processes sequence **left-to-right and right-to-left**.
   * Captures context from both past and future.
   * Output shape: (sequence\_length, 2 \* hidden\_dim)

3. **Dense Layer**:

   * Fully connected layer projecting to number of entity classes.

4. **Softmax Layer**:

   * Predicts **class probabilities per word**.
   * Output shape: (sequence\_length, num\_classes)

---

## 🧪 Training the Model

### Dataset

| Dataset           | Description                          |
| ----------------- | ------------------------------------ |
| **CoNLL-2003**    | Standard benchmark (English, German) |
| **OntoNotes 5.0** | Larger, multilingual NER dataset     |
| **WNUT 17**       | Emerging and unusual entity types    |

Each sample includes:

* Input sentence (tokenized).
* BIO-labeled entity tags per word.

---

### Loss Function

Since this is a **multi-class classification problem per time step**, we use:

**Categorical Cross-Entropy (token-wise)**:

$$
\mathcal{L} = -\sum_{t=1}^{T} \sum_{c=1}^{C} y_{t,c} \cdot \log(p_{t,c})
$$

* $T$: sequence length
* $C$: number of classes
* $y_{t,c}$: true label (one-hot)
* $p_{t,c}$: predicted probability

✅ Total loss is **averaged over all tokens** in the batch.

---

### Optimization

* **Optimizer**: Adam or RMSProp
* **Batching**: Pad sequences to uniform length
* **Masking**: Use masking to ignore padded positions during loss computation

---

## 🔍 Evaluation Metrics

| Metric        | Meaning                                  |
| ------------- | ---------------------------------------- |
| **Precision** | % of predicted entities that are correct |
| **Recall**    | % of actual entities that were found     |
| **F1-score**  | Harmonic mean of precision and recall    |

✅ Evaluated at the **entity-level**, not token-level:

* "B-PER I-PER" predicted correctly → 1 correct entity.
* One wrong token → the whole entity is incorrect.

---

## 💡 Enhancements to the RNN-Based NER

| Technique                      | Benefit                                                |
| ------------------------------ | ------------------------------------------------------ |
| **Character-level embeddings** | Capture morphology (e.g., “-ville” in cities)          |
| **CRF layer on top**           | Ensures valid label sequences (e.g., no I-LOC after O) |
| **Pretrained embeddings**      | Boost performance with GloVe, FastText, or BERT        |
| **Attention mechanism**        | Focus on relevant context tokens                       |
| **Dropout/Regularization**     | Prevent overfitting                                    |

---

## 🔐 Real-World Applications

* **Search engines** → Highlight key people, locations, companies.
* **Chatbots/assistants** → Understand user context ("Remind me to call Sarah").
* **Document indexing** → Extract structured info from contracts/emails.
* **Financial & legal analytics** → Identify entities in sensitive documents.

---

## ✅ Summary

| Component  | Description                                  |
| ---------- | -------------------------------------------- |
| Task       | Identify and classify named entities in text |
| Input      | Sequence of words/tokens                     |
| Model      | BiLSTM + Softmax (or BiLSTM + CRF)           |
| Output     | Label per word (e.g., B-PER, I-LOC, O)       |
| Loss       | Categorical cross-entropy per token          |
| Evaluation | F1 score over complete entities              |
| Datasets   | CoNLL-2003, OntoNotes, WNUT                  |
