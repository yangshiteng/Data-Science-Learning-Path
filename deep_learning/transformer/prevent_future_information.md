### 🔍 **Part 7.2: Masking – Preventing Future Information**

*(Also known as **causal masking** or **look-ahead masking**)*

---

### ✅ **Quick Summary:**

When training the decoder in a Transformer, we must prevent it from “cheating” by looking at future tokens it’s supposed to predict.
**Masking** ensures that each token can only **attend to itself and previous tokens**, not the ones that come after.

---

## 🎯 **Why is Masking Needed?**

Let’s say the target sequence is:

> **“Le chat s'est assis.”**

During training, the decoder input is:

> **[<s>, Le, chat, s', est]**

We want it to predict the next tokens:

> **[Le, chat, s', est, assis]**

But if attention isn’t masked, each decoder token could **access the full sequence**, including **the answer it's supposed to predict** — which makes learning trivial and unrealistic.

---

## 🚫 **No Masking (BAD)**

| Step            | Decoder can see                              |
| --------------- | -------------------------------------------- |
| Predict "chat"  | `<s>, Le, chat, s', est, assis` → ❌ cheating |
| Predict "assis" | full sentence → ❌ already sees the answer    |

---

## ✅ **With Masking (GOOD)**

| Step            | Decoder can see only     |
| --------------- | ------------------------ |
| Predict "chat"  | `<s>, Le`                |
| Predict "assis" | `<s>, Le, chat, s', est` |

Only **past and current tokens** are visible.
The model must **learn to predict the next token** based on **history**, not answers.

---

## 🧩 **How It Works (Under the Hood)**

The Transformer uses a **triangular mask matrix** in the attention mechanism.

For a sequence of length 5:

```plaintext
Mask =

[1 0 0 0 0]   <-- token 1 can only see itself  
[1 1 0 0 0]   <-- token 2 can see 1 and 2  
[1 1 1 0 0]   <-- token 3 can see 1, 2, 3  
[1 1 1 1 0]  
[1 1 1 1 1]
```

These 0s are converted to `-∞` in the attention score matrix (or a very large negative number like `-1e9`), so softmax assigns them a weight close to **0**.

---

### 🔧 **Types of Masks Used in Transformers**

| Mask Type               | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| **Look-ahead (causal)** | Prevent decoder from seeing future tokens      |
| **Padding mask**        | Prevent model from attending to `<PAD>` tokens |
| **Combined mask**       | Both look-ahead and padding at once            |

---

### 🧠 One-Liner Summary:

> Masking forces the decoder to **generate tokens step-by-step**, just like it would during inference — making training more realistic and robust.

---

Would you like to continue to **Part 7.3: Loss functions and optimization**, or would you prefer a **visual or code example** of how masking is applied?
