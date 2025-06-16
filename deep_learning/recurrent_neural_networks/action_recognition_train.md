## 🏃‍♂️📹 Action Recognition Using RNNs — Full Training Workflow

This training pipeline describes how to build an action recognition model using a CNN-RNN architecture on a real dataset like **UCF-101**.

---

### 🗂️ 1. Training Dataset

#### 📘 Dataset: UCF-101

* Contains **13,320 video clips** of human actions
* Covers **101 categories** like:

  * "Jumping Jack", "Biking", "Shooting Ball", "Walking with Dog"
* Each video is typically 1–10 seconds long, with **RGB frame sequences**

#### ✅ Labels:

* Each video is associated with a **single action label**
* These are **categorical labels** (e.g., “walking” → class ID 25)

---

### 📥 2. Input Data

Each training sample is a **video clip**, usually processed into:

* A fixed number of frames (e.g., 16 or 32)
* Each frame resized (e.g., to 224×224)
* Each frame passed through a CNN (e.g., ResNet50) to extract **spatial features**

#### Final input shape:

```
(batch_size, time_steps, feature_dim)
e.g., (32, 16, 2048) → 32 videos, 16 frames each, 2048-dim feature vector
```

---

### 📤 3. Output Data

The output for each video is a **class label** (e.g., “swinging” → class 17).

* Labels are **one-hot encoded** vectors for classification:

  ```
  Class 17 of 101 → [0, 0, ..., 1, ..., 0]
  ```

---

### 🧹 4. Data Preprocessing

#### 🔄 Frame Sampling:

* Uniformly sample 16 or 32 frames from each video
* Handle short videos via frame duplication or padding

#### 🖼 Frame Processing:

* Resize to CNN input size (e.g., 224×224)
* Normalize pixel values (e.g., scale to \[0, 1])

#### 🧠 Feature Extraction:

* Pass each frame through a **pretrained CNN**
* Extract intermediate features (e.g., 2048-dim from ResNet)
* Stack them into a temporal feature sequence

#### 🧾 Label Encoding:

* Convert string labels (e.g., “biking”) into integer class IDs
* Then one-hot encode for classification

---

### 🧠 5. Model Architecture

Typical architecture: **CNN + RNN + Classifier**

* **CNN** (e.g., ResNet50):

  * Extracts spatial features from each frame
  * Applied independently to each frame (preprocessing step)

* **RNN** (e.g., LSTM, GRU, BiLSTM):

  * Processes the **sequence of frame features**
  * Learns temporal dynamics (movement, action flow)

* **Dense + Softmax**:

  * Predicts the **probability distribution** over action classes

---

### 🧮 6. Loss Calculation

The model uses **categorical cross-entropy loss** for multi-class classification.

#### Formula:

```
L = – ∑ yᵢ · log(pᵢ)
```

Where:

* `yᵢ`: one-hot encoded true label
* `pᵢ`: predicted probability from softmax

Loss is averaged across the batch.

---

### 🏋️ 7. Training Process

1. **Prepare data batches**:

   * For each batch: sampled frame features + one-hot labels

2. **Forward pass**:

   * Sequence of features → RNN → dense layer → predicted action class

3. **Loss computation**:

   * Compare prediction with ground truth using cross-entropy

4. **Backpropagation**:

   * Gradients flow through softmax, RNN, and possibly CNN (if end-to-end)

5. **Optimization**:

   * Use Adam or RMSprop to update model weights

6. **Repeat**:

   * Over all training batches and multiple epochs

---

### 🔚 8. Model Output

For each input video sequence, the model outputs:

* A **vector of probabilities** over all action classes:

  ```
  [0.01, 0.03, ..., 0.95, ..., 0.01]
  ```

* The **predicted label** is the one with the highest probability:

  ```
  argmax(output) → class ID → class name
  ```

---

### ✅ Summary Table

| Step              | Description                                        |
| ----------------- | -------------------------------------------------- |
| **Dataset**       | UCF-101 or similar action-labeled videos           |
| **Input**         | Sequences of CNN frame features (e.g., 16 × 2048)  |
| **Output**        | One-hot encoded class labels                       |
| **Preprocessing** | Frame sampling, CNN features, label encoding       |
| **Model**         | RNN (LSTM/GRU) over CNN features + Dense + Softmax |
| **Loss**          | Categorical Cross-Entropy                          |
| **Output Format** | Probability vector → predicted class               |
