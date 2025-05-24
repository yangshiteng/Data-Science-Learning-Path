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

### 🔑 **Common Sampling Methods**

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
✅ Use `np.argmax(predicted_probs)` → greedy sampling.
✅ Use `np.random.choice(vocab, p=predicted_probs)` → random sampling.
✅ Limit to top-k → manually zero out others, then rescale and sample.
✅ Apply temperature → modify `predicted_probs` with the formula above before sampling.

---

### ✅ **Summary Table**

| Sampling Method | How It Works                                   | When To Use                   |
| --------------- | ---------------------------------------------- | ----------------------------- |
| Greedy          | Always pick most probable word                 | Coherence, correctness        |
| Random          | Sample proportional to predicted probabilities | Creativity, surprise          |
| Top-k           | Sample from top k candidates                   | Balanced creativity + safety  |
| Temperature     | Scale probability sharpness or softness        | Fine-tuned randomness control |
