### 🏗 **Step 1: Prepare the text data**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical, pad_sequences
import numpy as np

# Example dataset (small; you can replace with large text!)
data = """
cause baby now we got bad blood
you know it used to be mad love
so take a look what you have done
cause baby now we got bad blood
"""
```

✅ **What’s happening?**
We load a small block of song lyrics as our **raw text**.

In practice, you could load larger files (like full songbooks or novels).

---

---

### 🛠 **Step 2: Tokenize the text**

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1  # +1 for padding
```

✅ **What’s happening?**

* We create a **Tokenizer**, which will convert words to **integer indices**.
* For example:

  * “cause” → 1
  * “baby” → 2
  * “now” → 3
* We calculate the **vocabulary size** (`total_words`) — this will be used for the embedding layer and the output layer.

---

---

### 🏗 **Step 3: Create input–output pairs**

```python
input_sequences = []
for line in data.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
```

✅ **What’s happening?**

* We split the data **line by line**.
* For each line, we turn it into a sequence of word indices.
* We create **n-gram sequences**:

  * For “cause baby now”, we generate:

    * \[“cause”, “baby”] → “now”
    * \[“cause”, “baby”, “now”] → “we”

We now have a list of **growing sequences** ready for training.

---

---

### 🧩 **Step 4: Pad the sequences**

```python
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))
```

✅ **What’s happening?**

* Since sequences are of varying lengths, we **pad them** (usually with zeros at the start) so they all match the same length.
* This ensures that the input can fit into a fixed-size model.

Example:

| Raw sequence | After padding |
| ------------ | ------------- |
| \[1, 2]      | \[0, 0, 1, 2] |
| \[1, 2, 3]   | \[0, 1, 2, 3] |

---

---

### 🔀 **Step 5: Split input (X) and target (y)**

```python
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)
```

✅ **What’s happening?**

* `X`: All tokens **except the last one** (input sequence).
* `y`: The **next word** we want to predict.
* We one-hot encode `y` using `to_categorical`.

Example:

| Input (X)     | Target (y)  |
| ------------- | ----------- |
| \[0, 0, 1, 2] | 3 (→ “now”) |
| \[0, 1, 2, 3] | 4 (→ “we”)  |

---

---

### 🏗 **Step 6: Build the model**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=10, input_length=max_seq_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

✅ **What’s happening?**

* **Embedding layer** → Converts word indices into dense vectors of size 10.
* **LSTM layer** → Processes the sequence, capturing temporal dependencies.
* **Dense layer** → Outputs a softmax over all possible next words.

We compile the model with:

* **Loss**: Categorical cross-entropy (because it’s a multiclass classification problem).
* **Optimizer**: Adam.

---

---

### 🏋️ **Step 7: Train the model**

```python
history = model.fit(X, y, epochs=200, verbose=1)
```

✅ **What’s happening?**

* We train the model over 200 epochs.
* During each epoch:

  * It processes the full dataset.
  * It updates weights to minimize prediction error.
  * It outputs training accuracy and loss.

---

---

### ✨ **Step 8: Generate new text (sampling)**

```python
def generate_text(seed_text, next_words, model, max_seq_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_probs, axis=1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example usage
seed_text = "cause baby"
generated = generate_text(seed_text, 10, model, max_seq_len)
print("Generated text:")
print(generated)
```

✅ **What’s happening?**

* We start with a **seed text** (e.g., “cause baby”).
* We predict the next word, append it, and **feed the new sequence back in**.
* This **chains predictions** to generate longer, coherent phrases.

Example generated output:

```
cause baby now we got bad blood you know it used
```

---

---

### 🚀 **Optional Extensions**

* Use **beam search** instead of greedy sampling.
* Add **temperature scaling** to control randomness.
* Train on a **much larger dataset** (e.g., full lyrics, books, movie scripts).

---

---

### ✅ **Summary Table**

| Step          | What We Did                                          |
| ------------- | ---------------------------------------------------- |
| Prepare data  | Tokenized text, built sequences, padded inputs       |
| Build model   | Embedding + LSTM + Dense softmax                     |
| Train model   | Used cross-entropy loss, Adam optimizer              |
| Generate text | Seeded a start phrase, predicted next words stepwise |
