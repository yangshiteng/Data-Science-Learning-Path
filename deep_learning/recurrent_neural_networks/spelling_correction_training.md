## 🏗 **Complete Training Process: Spelling/Grammar Correction Using RNNs**

---

### 🌍 **Objective**

We aim to build a model that takes as input a **possibly incorrect sentence** and outputs its **corrected version**.

Example:

❌ Input → “She dont like apples.”

✅ Output → “She doesn’t like apples.”

This is formulated as a **sequence-to-sequence (seq2seq)** problem:

* Input: noisy or erroneous text.
* Output: clean, corrected text.

---

---

### 🛠 **Step 1: Training Dataset**

---

✅ **What kind of data do we need?**
We need a **parallel corpus**:

* Sentence pairs: (incorrect version, corrected version).

---

✅ **Where do we get this data?**

* **Public datasets**:

  * Lang-8 Learner Corpus (crowdsourced language learner mistakes).
  * CoNLL-2014 Shared Task (English grammar error correction).
  * JFLEG (fluency-oriented grammar corrections).
* **Synthetic data**:

  * Start with grammatically correct sentences.
  * Inject random or systematic errors (like typos, wrong verb forms, missing articles) to create incorrect inputs.

---

✅ **Example dataset entries**:

| Incorrect Sentence          | Corrected Sentence             |
| --------------------------- | ------------------------------ |
| “I go to school yesterday.” | “I went to school yesterday.”  |
| “He have a car.”            | “He has a car.”                |
| “She dont know the answer.” | “She doesn’t know the answer.” |
| “They is playing outside.”  | “They are playing outside.”    |

---

---

### 🛠 **Step 2: Data Preprocessing**

---

#### **1️⃣ Normalize the text**

* Lowercase all words (if case isn’t important).
* Remove or standardize punctuation.
* Optionally, handle contractions (e.g., “doesn’t” → “does not”).

---

#### **2️⃣ Tokenize**

* Decide the granularity:

  * **Character-level** → better for spelling correction.
  * **Word-level** → better for grammar correction.
* Build a vocabulary of tokens:

  * Map each word (or character) to a unique index.
  * Include special tokens like `<start>`, `<end>`, `<pad>`, `<unk>`.

---

#### **3️⃣ Convert to sequences**

* For each sentence, replace tokens with their indices.

Example (word-level):
“she dont know” → \[5, 23, 88]
“she doesn’t know” → \[5, 45, 88]

---

#### **4️⃣ Pad or truncate sequences**

* RNNs expect inputs of uniform length in each batch.
* Pad shorter sentences with `<pad>` tokens.
* Truncate overly long sentences to a maximum length.

---

#### **5️⃣ Prepare decoder targets**

* Decoder input → `<start>` + corrected sequence.
* Decoder target → corrected sequence + `<end>`.

This sets up the sequence-to-sequence prediction.

---

---

### 🧠 **Step 3: Build the RNN Model (Overview)**

---

✅ **Encoder**

* Takes the input (incorrect sentence).
* Embeds tokens into dense vectors.
* Processes the sequence using RNN layers (typically LSTM or GRU).
* Outputs hidden states summarizing the input.

✅ **Decoder**

* Takes encoder hidden states and `<start>` token.
* Generates the corrected sentence one token at a time.
* Uses softmax at each step to predict the next token.

✅ **Attention mechanism (optional)**

* Helps the decoder focus on specific parts of the input when generating each word.
* Improves performance, especially on longer sentences.

---

---

### 🏋 **Step 4: Define Loss Function**

---

✅ **What are we optimizing?**
We want the model to assign **high probability** to the correct next token at each decoding step.

✅ **Loss function used**:

* **Categorical cross-entropy** (if targets are one-hot encoded).
* **Sparse categorical cross-entropy** (if targets are integer indices).

---

#### **How is the loss calculated?**

At each time step $t$:

* Model predicts a probability distribution over the vocabulary.
* We look up the probability assigned to the correct next token $y_t$.
* Compute:

$$
\text{Loss}_t = -\log(P(y_t))
$$

---

✅ **Total loss for a sentence**:

* Average (or sum) the losses across all time steps.

✅ **Total loss for a batch**:

* Average over all sentences in the batch.

This loss guides the model to improve its predictions over time.

---

---

### 🏃 **Step 5: Train the Model**

---

For each training epoch:

1. **Feed input sequences** (incorrect sentences) to the encoder.
2. **Feed shifted target sequences** to the decoder (with teacher forcing).
3. **Predict next tokens** at each step.
4. **Compute cross-entropy loss** between predictions and true corrected tokens.
5. **Backpropagate errors** to update model weights.
6. Repeat over all batches and all epochs.

Example progress:

| Epoch | Training Loss | Validation Accuracy |
| ----- | ------------- | ------------------- |
| 1     | 2.1           | 52%                 |
| 5     | 1.3           | 68%                 |
| 10    | 0.9           | 75%                 |

---

---

### ✨ **Step 6: Inference (Correction at Test Time)**

---

✅ Provide a noisy input sentence.

✅ Use the encoder to process it into hidden states.

✅ Use the decoder to generate corrected tokens, one at a time, until the `<end>` token.

✅ Use greedy search or beam search to select the most likely sequence.

Example:

* Input → “He dont like it.”
* Output → “He doesn’t like it.”

---

---

### 📊 **Step 7: Evaluation**

---

✅ **Metrics used**:

* **Accuracy** → Percentage of exactly correct sentences.
* **Edit distance** → Number of insertions, deletions, substitutions to match the target.
* **BLEU score** → N-gram overlap between generated and reference corrections.
* **F0.5 score** → Precision-weighted measure from CoNLL grammar correction benchmarks.

✅ **Qualitative checks**:

* Read generated corrections.
* Spot-check fluency and correctness.

---

---

### 🚀 **Applications**

✅ Writing assistants (e.g., Grammarly, Microsoft Editor).

✅ Language learning apps (helping learners fix grammar).

✅ Automated text cleaning for chatbots, customer support logs, search queries.

---

---

### ✅ Summary Table

| Step          | Description                                                |
| ------------- | ---------------------------------------------------------- |
| Dataset       | Pairs of noisy + corrected sentences.                      |
| Preprocessing | Normalize, tokenize, map to indices, pad sequences.        |
| Model         | Encoder–decoder RNN with attention (optional).             |
| Loss Function | Cross-entropy over predicted vs. correct tokens.           |
| Training      | Optimize weights over batches and epochs to minimize loss. |
| Evaluation    | Measure accuracy, edit distance, BLEU, F0.5.               |
