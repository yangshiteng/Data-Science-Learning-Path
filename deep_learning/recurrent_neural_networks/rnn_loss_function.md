## 🎯 **Loss Functions for Sequential Tasks**

### 🧠 **What Are Sequential Tasks?**

**Sequential tasks** are problems where the input and/or output is a **sequence**, and **order matters**.
Examples include:

* Language modeling (predicting the next word)
* Machine translation
* Speech recognition
* Time series forecasting

To train models like RNNs, LSTMs, GRUs, or Transformers on these tasks, we need to measure **how well the model’s predictions match the expected sequence** — and that’s where **loss functions** come in.

---

## 📉 **Why Loss Functions Matter**

A **loss function** computes the **difference between predicted outputs and ground truth labels**.
During training, models **minimize this loss** to improve performance.

---

## 🔍 **Types of Loss Functions for Sequential Tasks**

### 1. 🧾 **Cross-Entropy Loss**

#### ✅ Use when: **Each step outputs a classification (e.g., a word or token)**

Used in:

* Language modeling
* Translation
* Text generation

Formula (for a single time step):

$$
\mathcal{L}_{t} = -\sum_{i} y_{t}^{(i)} \log(\hat{y}_{t}^{(i)})
$$

Where:

* $y_t$: One-hot encoded true distribution at time $t$
* $\hat{y}_t$: Model’s predicted probability distribution

If you have a sequence of length $T$, the total loss is:

$$
\mathcal{L}_{\text{total}} = \sum_{t=1}^{T} \mathcal{L}_{t}
$$

---

### 2. 🔢 **Mean Squared Error (MSE)**

#### ✅ Use when: **Output is continuous or numerical**

Used in:

* Time series forecasting
* Signal prediction
* Sensor data modeling

Formula:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)^2
$$

Minimizes the squared difference between predicted and actual values at each time step.

---

### 3. 📈 **Mean Absolute Error (MAE)**

#### ✅ Use when: **You want a more robust loss than MSE (less sensitive to outliers)**

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{T} \sum_{t=1}^{T} |y_t - \hat{y}_t|
$$

Used similarly to MSE, often in regression-based sequence tasks.

---

### 4. 🧠 **CTC Loss (Connectionist Temporal Classification)**

#### ✅ Use when: **Input and output sequence lengths differ and alignment is unknown**

Used in:

* Speech recognition
* Handwriting recognition

CTC allows models to learn from **unsegmented data**, where the alignment between input frames and target labels isn't given.

---

### 5. 🎯 **Sequence-Level Losses (Task-Specific)**

#### Used in: **End-to-end sequence predictions**

Examples:

* **BLEU loss** (used in machine translation evaluation)
* **ROUGE loss** (used in summarization)
* **Custom reinforcement-based losses** (like REINFORCE for sequence generation)

These are typically used in advanced setups where the entire output sequence is evaluated as a unit (e.g., translation quality), rather than step-by-step.

---

## 🧾 Summary Table

| Loss Function  | Use Case                               | Output Type       |
| -------------- | -------------------------------------- | ----------------- |
| Cross-Entropy  | Classification (e.g., word prediction) | Discrete          |
| MSE (L2 Loss)  | Time series, regression tasks          | Continuous        |
| MAE (L1 Loss)  | Robust regression                      | Continuous        |
| CTC Loss       | Unaligned sequences                    | Discrete labels   |
| Sequence-Level | End-to-end evaluation (BLEU, etc.)     | Structured output |
