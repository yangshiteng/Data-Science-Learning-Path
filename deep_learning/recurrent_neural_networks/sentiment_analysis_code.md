### 🛠 **Step 1: Import Libraries**

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```
✅ **Explanation:** We import TensorFlow, load IMDb data, handle sequence padding, and bring in model layers.

---

### 🛠 **Step 2: Load the Dataset**

```python
num_words = 10000  # use top 10,000 words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
```
✅ **Explanation:** We load IMDb reviews, where each review is already tokenized as a sequence of word indices, limited to the 10,000 most frequent words.

---

### 🛠 **Step 3: Preprocess the Data**

```python
maxlen = 200  # maximum review length
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
```
✅ **Explanation:** We pad (or truncate) all sequences to length 200 so they can be processed in uniform batches.

---

### 🛠 **Step 4: Build the RNN Model**

```python
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=128, input_length=maxlen))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
```
✅ **Explanation:**
- **Embedding layer:** Turns word indices into 128-dimensional vectors.
- **LSTM layer:** Processes the sequence, capturing temporal patterns.
- **Dense layer:** Outputs a probability (between 0–1) for binary sentiment.

---

### 🛠 **Step 5: Compile the Model**

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
✅ **Explanation:**
- **Optimizer:** Adam adjusts learning rates adaptively.
- **Loss:** Binary cross-entropy is used for two-class (positive/negative) classification.
- **Metrics:** We track accuracy during training.

---

### 🛠 **Step 6: Train the Model**

```python
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)
```
✅ **Explanation:**
- We train for 5 epochs.
- Use a batch size of 64 for gradient updates.
- Set aside 20% of the training data for validation.

---

### 🛠 **Step 7: Evaluate on Test Set**

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc:.3f}')
```
✅ **Explanation:** We evaluate the trained model on the separate test set to check how well it generalizes.

---

### 🏋 **Loss Calculation in Detail**

For each batch:
- For binary labels \( y \) (0 or 1) and predicted probability \( p \):
\[
\text{Binary Cross-Entropy} = -[y \log(p) + (1 - y) \log(1 - p)]
\]
- The model minimizes the **average loss** over all samples in the batch.

✅ If the predicted \( p \) is close to the true label \( y \), the loss is small.  
✅ If the prediction is wrong (e.g., \( p = 0.9 \) for \( y = 0 \)), the loss is large, pushing the model to adjust weights.

---

### ✅ Final Summary

| Step             | Details                                                         |
|------------------|----------------------------------------------------------------|
| Dataset         | IMDb reviews (binary sentiment)                                |
| Preprocessing   | Pad sequences to fixed length                                  |
| Model          | Embedding → LSTM → Dense + sigmoid                             |
| Loss Function  | Binary cross-entropy                                           |
| Training       | Optimize weights over batches, track validation accuracy        |
| Evaluation     | Measure final accuracy on test set                             |
