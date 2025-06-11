## 🧠🎯 What is NER?

**Named Entity Recognition (NER)** is a sequence labeling task where the model must recognize and categorize entities in text, such as:

* **PERSON** → e.g., “Marie Curie”
* **LOCATION** → e.g., “New York”
* **ORGANIZATION** → e.g., “United Nations”
* **DATE**, **MONEY**, **EVENT**, etc.

---

## 🗂️ Step 1: Real-World Dataset

We use the **CoNLL-2003** dataset, a standard benchmark for English NER.

### 🧾 Dataset Format

Each sample is a sentence, where **each word is labeled** with its entity class using the **BIO tagging scheme**:

* `B-XXX`: Beginning of an entity of type XXX
* `I-XXX`: Inside that same entity
* `O`: Outside any entity

**Example sentence**:

```
Sentence:  Barack Obama visited Paris in July.
```

**Labeled as**:

| Token   | Label  |
| ------- | ------ |
| Barack  | B-PER  |
| Obama   | I-PER  |
| visited | O      |
| Paris   | B-LOC  |
| in      | O      |
| July    | B-DATE |
| .       | O      |

Each line in the raw dataset typically includes:

```
Token   POS   Chunk   NER
Barack  NNP   B-NP    B-PER
Obama   NNP   I-NP    I-PER
...     ...   ...     ...
```

---

## 🧹 Step 2: Data Preprocessing

NER requires the data to be processed into **sequences of tokens and labels**, suitable for RNN input.

### 2.1 Tokenization

* Each sentence is split into tokens (words or subwords).
* Tokens are indexed with a **vocabulary mapping**: e.g., “Paris” → ID 1043.

### 2.2 Label Encoding

* Convert tags into integers: e.g., `B-PER → 0`, `I-PER → 1`, `O → 2`, etc.
* BIO tag space typically contains 10–20 unique labels.

### 2.3 Sequence Padding

* Sentences are padded to the same length for batching.
* Masking is applied so padding doesn’t affect learning.

### 2.4 Word Embeddings (optional but common)

* Map tokens to pretrained word vectors (like GloVe or FastText).
* Each token is then represented as a vector (e.g., 100 or 300 dimensions).

---

## 📥 Step 3: Model Input and Output

### Input to the Model:

* A **batch of token sequences**, e.g., shape: `(batch_size, sequence_length)`
* Each token is represented by an **embedding vector**

### Output from the Model:

* A **sequence of class predictions**, one for each token
* Final output shape: `(batch_size, sequence_length, num_entity_classes)`

  * Example: if 9 entity tags, output is a softmax distribution over 9 classes **per token**

---

## 🔁 Step 4: RNN-based Model Structure

1. **Embedding Layer**
   Converts token IDs into dense vectors.

2. **BiLSTM Layer**
   Processes the token sequence in both directions (left-to-right and right-to-left) to capture full context.

3. **Dense (Linear) Layer**
   Maps hidden state at each timestep to a class score vector.

4. **Softmax**
   Computes probability distribution over entity classes for each token.

---

## 🧮 Step 5: Loss Function

The model is trained using **categorical cross-entropy loss**, applied **per token**.

### For a sentence of length `T`, the loss is:

$$
\text{Loss} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{c=1}^{C} y_{t,c} \cdot \log(p_{t,c})
$$

Where:

* $y_{t,c}$ is the true one-hot label at timestep $t$
* $p_{t,c}$ is the predicted probability for class $c$
* $C$ is the number of entity classes

✅ The loss is typically averaged across the sequence and batch.

---

## 🧠 Step 6: Training Process

For each training epoch:

1. Sample a batch of tokenized, padded sentences and corresponding label sequences.
2. Forward pass:

   * Embeddings → BiLSTM → Dense → Softmax
3. Compute per-token loss using categorical cross-entropy.
4. Backpropagate and update model weights using Adam or RMSProp.
5. Repeat for all batches.

---

## 📊 Step 7: Model Evaluation

The trained model is evaluated using **precision, recall, and F1-score** based on exact entity matches.

✅ Entity-level metrics are used (not just individual tokens).

### Example:

Predicted: `B-PER I-PER`
Ground truth: `B-PER I-PER` → ✅ Correct

Predicted: `B-PER O`
Ground truth: `B-PER I-PER` → ❌ Incorrect entity span

---

## ✅ Summary Table

| Component         | Description                                       |
| ----------------- | ------------------------------------------------- |
| Dataset           | CoNLL-2003 (BIO-tagged sentences)                 |
| Input             | Sequence of word IDs or embeddings                |
| Output            | Sequence of entity class predictions (per word)   |
| Preprocessing     | Tokenization, label encoding, padding, embeddings |
| Model             | Embedding → BiLSTM → Dense → Softmax              |
| Loss Function     | Token-wise categorical cross-entropy              |
| Evaluation Metric | F1 score (entity-level)                           |
