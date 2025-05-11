## 📏 **Perplexity in Language Modeling**

### 🧠 **What Is Perplexity?**

**Perplexity** is a measurement of **how well a language model predicts a sequence of words**.
It quantifies the model's **uncertainty** when predicting the next word in a sentence.

---

### 🔍 **Intuition**

> A good language model assigns **high probabilities** to the correct next words.
> A poor model assigns **low or scattered probabilities**, meaning it’s "perplexed."

So, **lower perplexity = better model**.

---

## 🧮 **How Is Perplexity Calculated?**

![image](https://github.com/user-attachments/assets/f923bf19-1671-44e9-a109-f944c0ad4d54)

---

### 🧠 Example

Suppose your model is predicting the next word in the sentence:
**"I am going to the"** → and the next word is **"store"**

* Good model: $P(\text{"store"}) = 0.8$ → low perplexity
* Bad model: $P(\text{"store"}) = 0.1$ → high perplexity

If the model is **perfect**, perplexity = 1 (no uncertainty)
If it guesses **randomly**, perplexity is **equal to the vocabulary size**.

---

## 📊 **How to Interpret Perplexity**

| Perplexity Value  | Meaning                          |
| ----------------- | -------------------------------- |
| \~1               | Perfect model (rare in practice) |
| 10–100            | Decent to strong language models |
| 500+              | Weak or untrained model          |
| ≈ Vocabulary Size | Random guessing                  |

---

## 📦 **When to Use Perplexity**

✅ Use it for:

* **Evaluating language models** (RNN, LSTM, GRU, Transformer)
* **Comparing different models** on the same dataset
* **Monitoring training/validation performance**

❌ Don’t use it if:

* You're evaluating **task-specific output** (e.g., translation, summarization) → use BLEU, ROUGE instead

---

## 🧾 Summary

| Feature          | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| What it measures | How confident the model is in predicting sequences            |
| Lower is better? | ✅ Yes                                                         |
| Type             | Probabilistic, exponential of average negative log-likelihood |
| Good for         | RNNs, LSTMs, GRUs, Transformers in language modeling          |
