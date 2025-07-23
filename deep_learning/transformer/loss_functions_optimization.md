### 🔍 **Part 7.3: Loss Functions and Optimization in Transformers**

---

### ✅ **Quick Summary:**

During training, Transformers learn to **predict the next token** in a sequence.
To measure how well the predictions match the expected output, we use a **loss function** — specifically, **cross-entropy loss** — along with **optimization algorithms** like Adam to update the model’s weights.

---

## 🧠 **1. Loss Function: Cross-Entropy**

### 📌 What is it?

A standard loss function for classification problems — measures the difference between the predicted probability distribution and the actual (true) distribution.

### 📐 Formula (for one token):

$$
\text{Loss} = -\log(p_{\text{true token}})
$$

### 🧾 Example:

Suppose the model predicts probabilities for vocabulary tokens:

```plaintext
Output: [“Le”, “chat”, “mange”, “s’est”, “assis”]
Predicted: [0.7, 0.2, 0.05, 0.03, 0.02]
True token: “Le” → index 0
Loss = -log(0.7) ≈ 0.357
```

You compute this for **every token in the sequence**, and take the **average**.

---

### 🔧 **2. Padding Masking in the Loss**

Many sequences are padded to match lengths in a batch.
We **ignore `<PAD>` tokens** when computing the loss — they don’t carry learning signal.

### ✔️ Solution:

* Create a **padding mask**
* Multiply the loss for each token by the mask:

  * 1 if it's a real token
  * 0 if it's padding
* Then take the mean over the non-zero positions

---

## ⚙️ **3. Optimization**

Transformers use:

* **Adam or AdamW** optimizer
* **Learning rate warm-up and decay** for stable training
* **Gradient clipping** to prevent exploding gradients (especially with big models)

---

### 🔁 **Training Steps:**

1. Forward pass:

   * Inputs go through encoder + decoder
   * Get output logits (unnormalized predictions)

2. Compute loss:

   * Apply softmax
   * Compare predictions to true tokens using cross-entropy
   * Mask padding tokens

3. Backward pass:

   * Compute gradients using backpropagation

4. Update weights:

   * Use Adam optimizer to update parameters

---

### 📊 Summary Table:

| Step             | Tool/Method                       |
| ---------------- | --------------------------------- |
| Prediction       | Softmax on decoder output         |
| Loss function    | Cross-entropy                     |
| Ignore padding   | Padding mask                      |
| Optimizer        | Adam / AdamW                      |
| Stability tricks | Warm-up, decay, gradient clipping |

---

### 🧠 One-Liner Summary:

> Transformers are trained using **cross-entropy loss** over the predicted tokens, with **masking for padding**, and **Adam optimization** to adjust the weights step-by-step.
