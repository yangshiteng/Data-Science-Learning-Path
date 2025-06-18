### 🔍 **What Does Each Attention Head Learn?**

---

### ✅ **Short Answer:**

Each attention head in a Transformer learns to **specialize** in different types of patterns or relationships in the data. This gives the model **multiple perspectives** on the same input — like a team of specialized readers.

---

### 🧠 **How Heads Learn Specialization**

Each attention head has its own set of **learnable projection matrices** for:

* Queries $W_i^Q$
* Keys $W_i^K$
* Values $W_i^V$

Because these matrices are different across heads, each head transforms and views the input **differently**. During training, each head naturally learns to focus on **distinct patterns** that help reduce the loss function.

---

### 🧪 **Examples of What Heads Learn (Empirically Observed)**

Researchers (like in the BERTology studies) have visualized and analyzed attention heads. Here are some typical roles:

| **Head Function**                 | **What It Might Attend To**                          |
| --------------------------------- | ---------------------------------------------------- |
| **Coreference resolution**        | Links pronouns to nouns (e.g., “he” → “the doctor”)  |
| **Syntactic roles**               | Connects subject and verb (e.g., “cat” ↔ “sat”)      |
| **Entity recognition**            | Focuses on named entities like people, places, dates |
| **Positional patterns**           | Attends to preceding or following words              |
| **Long-range dependency capture** | Connects distant but related words                   |
| **Punctuation handling**          | Helps segment sentences or clauses                   |

---

### 🔍 **Visual Example (from BERT studies)**

In this sentence:

> *"The trophy doesn’t fit in the suitcase because it is too small."*

Some heads may focus on:

* “it” → “trophy”
* “too small” → “suitcase”
* “fit” → both “trophy” and “suitcase”

Different heads are pulling **different dependency strings**.

---

### 🧠 Heads Aren’t Preassigned Roles

* There’s **no manual assignment** like “Head 1 must do grammar”.
* Instead, during training, the model **learns** which patterns reduce error best.
* Some heads may even learn **redundant patterns**, which still adds robustness.

---

### 📚 Takeaway:

> **Each head acts as a unique lens**, capturing relationships like grammar, meaning, or position. Together, they let the Transformer build a deep and flexible understanding of input data.
