### 🛠 **Step 1: Load Libraries and Dataset**

```python
import tensorflow as tf
import numpy as np
```
✅ **Explanation:** We import TensorFlow and NumPy for model building and array operations.

```python
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print(f'Length of text: {len(text)} characters')
```
✅ **Explanation:** We download and load the full Shakespeare dataset into memory.

---

### 🛠 **Step 2: Preprocess the Data**

```python
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
```
✅ **Explanation:** We create a vocabulary of unique characters, map characters to indices (`char2idx`), and convert the entire text to integer indices (`text_as_int`).

```python
seq_length = 100
sequences = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length + 1, drop_remainder=True)
```
✅ **Explanation:** We split the text into sequences of 101 characters (100 inputs + 1 target) to train the model to predict the next character.

```python
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
```
✅ **Explanation:** For each chunk, we create an input sequence and a target sequence, offset by one character.

```python
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True))
```
✅ **Explanation:** We shuffle the dataset and batch it for training.

---

### 🛠 **Step 3: Build the Model**

```python
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
```
✅ **Explanation:** We define a sequential model with:
- Embedding layer → transforms indices to dense vectors.
- LSTM layer → captures sequence dependencies.
- Dense layer → outputs logits over the vocabulary.

---

### 🛠 **Step 4: Compile the Model**

```python
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)
```
✅ **Explanation:** We use sparse categorical cross-entropy loss (since targets are integer indices) and Adam optimizer.

---

### 🛠 **Step 5: Train the Model**

```python
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = checkpoint_dir + '/ckpt_{epoch}'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

EPOCHS = 20
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
```
✅ **Explanation:** We train the model for 20 epochs and save checkpoints after each epoch.

---

### 🛠 **Step 6: Set Up Inference (Text Generation)**

```python
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
```
✅ **Explanation:** We rebuild the model for batch size 1 and load the saved weights.

---

### 🛠 **Step 7: Generate Text**

```python
def generate_text(model, start_string, num_generate=1000, temperature=1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    model.reset_states()
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :] / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)
```
✅ **Explanation:** We define a function that:
- Takes a starting string.
- Predicts one character at a time.
- Samples from the probability distribution (adjustable via `temperature`).
- Returns the generated text.

---

### 🛠 **Step 8: Run Text Generation**

```python
print(generate_text(model, start_string='To be, or not to be, ', num_generate=500))
```
✅ **Explanation:** We generate and print 500 new characters, starting from the given prompt.

---

### ✅ Summary
- Load and preprocess Shakespeare’s dataset.
- Build an embedding + LSTM + dense model.
- Train on next-character prediction.
- Save checkpoints.
- Generate new text using the trained model and sampling.
