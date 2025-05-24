## 🌍 **Example: Training an RNN Language Model on Song Lyrics**

Let’s say you want to train a model on a dataset of song lyrics to **generate new lyrics**.

---

### 📦 **Step 1: Choose Your Dataset**

We need a dataset.
For example, let’s use **Taylor Swift lyrics** collected from online sources.

We combine all her song lyrics into one big text file.

---

Example snippet:

```
Cause baby now we got bad blood
You know it used to be mad love
```

---

### 🛠 **Step 2: Preprocess the Data**

---

✅ **Tokenize the text**

Decide: do we want to model **character-level** or **word-level**?

* **Character-level**: Sequence → \['C', 'a', 'u', 's', 'e', ' ', 'b', 'a', 'b', 'y']
* **Word-level**: Sequence → \['Cause', 'baby', 'now', 'we', 'got', 'bad', 'blood']

For this example, let’s do **word-level**.

---

✅ **Build vocabulary**

Extract all unique words and assign each one an index.

Example:

| Word    | Index |
| ------- | ----- |
| ‘Cause’ | 0     |
| ‘baby’  | 1     |
| ‘now’   | 2     |
| ‘we’    | 3     |
| ‘got’   | 4     |
| …       | …     |

Let’s say we have 10,000 unique words (vocabulary size).

---

✅ **Create input–target pairs**

We slide a **window** over the text.

For a window size of 5, example:

* Input: \[‘Cause’, ‘baby’, ‘now’, ‘we’, ‘got’] → Target: ‘bad’
* Input: \[‘baby’, ‘now’, ‘we’, ‘got’, ‘bad’] → Target: ‘blood’

This forms our training dataset.

---

### 🧠 **Step 3: Build the RNN Model**

---

✅ **Input layer**

* Convert word indices to dense vectors using an **embedding layer**.

For example, 10,000 vocab → 128-dimensional embeddings.

---

✅ **RNN layer**

* Use **LSTM** or **GRU** to process the input sequence.

Example:

```python
lstm = LSTM(256, return_sequences=False)
```

This outputs a hidden state summarizing the input sequence.

---

✅ **Output layer**

* Apply a **dense (fully connected) layer** + **softmax** to predict the next word.

Example:

```python
dense = Dense(vocab_size, activation='softmax')
```

---

✅ **Model summary**

```
Input (5 words) → Embedding → LSTM → Dense → Next word probability
```

---

### 🏋️ **Step 4: Define Loss Function and Optimizer**

---

✅ **Loss function**:
Use **categorical cross-entropy** (comparing predicted vs. actual next word).

✅ **Optimizer**:
Use **Adam** or **SGD** for efficient weight updates.

---

### 🔁 **Step 5: Train the Model**

We loop over the dataset for multiple **epochs**.

At each batch:

1. Forward pass:

   * Input → prediction (probabilities over next word).
2. Compute loss:

   * Compare prediction vs. true next word.
3. Backward pass:

   * Calculate gradients (using backpropagation through time, BPTT).
4. Update weights:

   * Apply optimizer to adjust model weights.

---

✅ **Epoch example**

| Epoch | Average Loss |
| ----- | ------------ |
| 1     | 3.2          |
| 5     | 2.1          |
| 10    | 1.4          |

As epochs go up, loss typically goes down — the model gets better at predicting.

---

### 📊 **Step 6: Evaluate the Model**

---

✅ **Quantitative metrics**

* **Perplexity** → Measures how well the model predicts unseen text (lower is better).

✅ **Qualitative check**

* Generate some **sample lyrics** and see if they look coherent.

---

### 🔮 **Step 7: Generate New Lyrics**

---

✅ **Sampling process (generation)**

1. Provide a starting seed: e.g., “baby now we”
2. Predict next word: model suggests ‘got’
3. Append: “baby now we got”
4. Feed back in: “now we got” → predict → ‘bad’
5. Append: “baby now we got bad”
6. Continue for N words to generate a whole line or verse.

---

✅ **Sampling methods**

* **Greedy search**: Always pick the top prediction.
* **Top-k sampling**: Randomly sample from the top k predictions.
* **Temperature sampling**: Adjust randomness in predictions (higher temperature = more creative).

---

### ⚠️ **Step 8: Handle Challenges**

| Challenge                    | Solution                              |
| ---------------------------- | ------------------------------------- |
| Overfitting on training data | Use dropout or early stopping         |
| Vanishing gradients          | Use LSTM or GRU instead of simple RNN |
| Large vocabulary size        | Use techniques like sampled softmax   |
| Long-range dependencies      | Consider adding attention mechanisms  |

---

### 🌟 **What is Sampling in Language Models?**

When your model generates text, it doesn’t directly **write out words** — it produces a **probability distribution** over the entire vocabulary for the next word.

For example, after the phrase:

> “the cat sat on”

the model might predict:

| Word     | Probability |
| -------- | ----------- |
| “the”    | 0.05        |
| “mat”    | 0.60        |
| “rug”    | 0.25        |
| “chair”  | 0.05        |
| “banana” | 0.01        |

We need a **sampling method** to **choose** which word to pick next from this distribution.

---

## 🔑 **Common Sampling Methods**

---

### 1️⃣ **Greedy Sampling**

* Always pick the **word with the highest probability**.

From above, we would choose:

> “mat” (60%)

✅ **Pros**:

* Simple and fast.
* Often produces fluent, high-confidence sentences.

⚠ **Cons**:

* Can get repetitive or stuck in loops.
* Misses out on creative or diverse options.

---

### 2️⃣ **Random Sampling**

* **Randomly sample** the next word **proportional to its probability**.

Example:

* 60% → “mat” → high chance.
* 25% → “rug” → moderate chance.
* 1% → “banana” → very small chance (but not zero!).

✅ **Pros**:

* Introduces variability and creativity.
* Can generate surprising or novel outputs.

⚠ **Cons**:

* Might pick low-probability (nonsensical) words.
* Risk of lower overall coherence.

---

### 3️⃣ **Top-k Sampling**

* First, **narrow down to the top k most probable words**.
* Then randomly sample **only from this shortlist**.

Example with k=2:

| Top words | Probability |
| --------- | ----------- |
| “mat”     | 0.60        |
| “rug”     | 0.25        |

Randomly sample between “mat” and “rug” (ignore others).

✅ **Pros**:

* Balances between greedy and random sampling.
* Prevents strange, low-probability words from sneaking in.

⚠ **Cons**:

* Needs you to tune **k** carefully.
* Still might become repetitive if k is too small.

---

### 4️⃣ **Temperature Sampling**

* **Adjust the sharpness or softness** of the probability distribution.

Formula:

$$
P_i^{\text{new}} = \frac{P_i^{1/T}}{\sum_j P_j^{1/T}}
$$

Where:

* $T$ = temperature.

  * **T < 1** → sharpen probabilities (more confident, greedy-like).
  * **T > 1** → flatten probabilities (more exploratory, random).

Example:

* At **T = 0.7**, the model focuses on top words.
* At **T = 1.5**, even rare words get boosted.

✅ **Pros**:

* Flexible control over randomness and creativity.
* Can make outputs more diverse without going totally random.

⚠ **Cons**:

* Needs experimentation to pick the right **T**.

---

### 🛠 **How Do We Apply These?**

In code, after:

```python
predicted_probs = model.predict(token_list)
```

You can:
* ✅ Use `np.argmax(predicted_probs)` → greedy sampling.
* ✅ Use `np.random.choice(vocab, p=predicted_probs)` → random sampling.
* ✅ Limit to top-k → manually zero out others, then rescale and sample.
* ✅ Apply temperature → modify `predicted_probs` with the formula above before sampling.

---

### ✅ **Summary Table**

| Sampling Method | How It Works                                   | When To Use                   |
| --------------- | ---------------------------------------------- | ----------------------------- |
| Greedy          | Always pick most probable word                 | Coherence, correctness        |
| Random          | Sample proportional to predicted probabilities | Creativity, surprise          |
| Top-k           | Sample from top k candidates                   | Balanced creativity + safety  |
| Temperature     | Scale probability sharpness or softness        | Fine-tuned randomness control |

---

## ✅ **Summary of Full Training Process**

| Step          | Details                                      |
| ------------- | -------------------------------------------- |
| Dataset       | Collect and clean large text corpus          |
| Preprocessing | Tokenize, build vocabulary, create sequences |
| Model         | Embedding + LSTM/GRU + Dense (softmax)       |
| Training loop | Forward pass → loss → backprop → update      |
| Evaluation    | Check loss, perplexity, generate text        |
| Sampling      | Use trained model to generate new sequences  |
