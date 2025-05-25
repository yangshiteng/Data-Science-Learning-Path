### 🛠 **Step 1: Import Libraries and Prepare the Dataset**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
import numpy as np
```
✅ **Explanation:** We import TensorFlow, Keras components, and NumPy. These libraries handle text processing, sequence padding, neural network modeling, and array operations.

We will use **tiny example data** (in real use, you'd load a large dataset):

```python
articles = [
    "the us economy grew at an annual rate of 3.2 percent in q1 driven by consumer spending",
    "scientists discover new species of bird in the amazon rainforest",
    "the stock market reached a new high today amid economic optimism"
]

summaries = [
    "us economy grows 3.2 percent",
    "new bird species found in amazon",
    "stock market hits new high"
]
```
✅ **Explanation:** We define three article-summary pairs.

---

### 🛠 **Step 2: Tokenize and Preprocess Text**

```python
article_tokenizer = Tokenizer()
article_tokenizer.fit_on_texts(articles)
article_vocab_size = len(article_tokenizer.word_index) + 1

summary_tokenizer = Tokenizer()
summary_tokenizer.fit_on_texts(summaries)
summary_vocab_size = len(summary_tokenizer.word_index) + 1
```
✅ **Explanation:** We create tokenizers for both articles and summaries, building a word-to-index map. We also calculate the vocabulary size for each.

```python
article_sequences = article_tokenizer.texts_to_sequences(articles)
summary_sequences = summary_tokenizer.texts_to_sequences(summaries)
```
✅ **Explanation:** We convert the text data into sequences of integer indices.

```python
max_article_len = max(len(seq) for seq in article_sequences)
max_summary_len = max(len(seq) for seq in summary_sequences)
```
✅ **Explanation:** We find the longest sequence length for articles and summaries to know how much to pad.

```python
encoder_input_data = pad_sequences(article_sequences, maxlen=max_article_len, padding='post')
decoder_input_data = pad_sequences(summary_sequences, maxlen=max_summary_len, padding='post')
```
✅ **Explanation:** We pad all sequences to the same length so they can be fed into the neural network.

```python
decoder_target_data = np.zeros((len(summaries), max_summary_len, summary_vocab_size), dtype='float32')
for i, seq in enumerate(summary_sequences):
    for t, word_idx in enumerate(seq):
        if t > 0:
            decoder_target_data[i, t - 1, word_idx] = 1.0
```
✅ **Explanation:** We prepare the target data as a one-hot encoded array, shifted by one step to train the model to predict the next word.

---

### 🛠 **Step 3: Build the Encoder-Decoder Model**

```python
embedding_dim = 50
latent_dim = 100
```
✅ **Explanation:** We set the embedding dimension (word vector size) and latent dimension (LSTM hidden state size).

```python
encoder_inputs = Input(shape=(max_article_len,))
enc_emb = Embedding(article_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]
```
✅ **Explanation:** The encoder reads the padded article sequences, embeds them, passes them through an LSTM, and outputs the final hidden and cell states.

```python
decoder_inputs = Input(shape=(max_summary_len,))
dec_emb_layer = Embedding(summary_vocab_size, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(summary_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
```
✅ **Explanation:** The decoder takes the summary input, embeds it, runs through an LSTM (initialized with encoder states), and outputs a probability distribution over the summary vocabulary at each time step.

```python
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```
✅ **Explanation:** We compile the encoder–decoder model using categorical cross-entropy loss (to match predicted vs. true words) and print its summary.

---

### 🛠 **Step 4: Train the Model**

```python
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=2,
    epochs=100,
    verbose=1
)
```
✅ **Explanation:** We train the model, feeding in both encoder and decoder inputs, and optimizing against the target decoder outputs.

---

### 🛠 **Step 5: Build Inference Models for Generation**

```python
encoder_model = Model(encoder_inputs, encoder_states)
```
✅ **Explanation:** We create a model that encodes new articles and returns the final encoder states.

```python
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)
```
✅ **Explanation:** We define the decoder model separately for inference — it takes a word + previous states and outputs the next word + updated states.

---

### 🛠 **Step 6: Define Summary Generation Function**

```python
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = summary_tokenizer.word_index['start'] if 'start' in summary_tokenizer.word_index else 1

    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_word = None
        for word, index in summary_tokenizer.word_index.items():
            if sampled_token_index == index:
                sampled_word = word
                break
        if sampled_word is None:
            break

        decoded_sentence += ' ' + sampled_word

        if sampled_word == 'end' or len(decoded_sentence.split()) > max_summary_len:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence
```
✅ **Explanation:** This function generates a summary by looping:
- Predict the next word.
- Append it to the output.
- Feed it back in for the next step.
- Stop when we hit `<end>` or max length.
