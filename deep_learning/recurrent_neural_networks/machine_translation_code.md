### 🛠 **Step 1: Import Libraries**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```
✅ **Explanation:** We import libraries for modeling, text preprocessing, and sequence handling.

---

### 🛠 **Step 2: Prepare Sample Data**

```python
english_sentences = ['i am happy', 'how are you', 'thank you', 'i love travel']
french_sentences = ['je suis heureux', 'comment ça va', 'merci', "j'aime voyager"]
```
✅ **Explanation:** We define small English-French sentence pairs (in practice, you'd load a large dataset).

---

### 🛠 **Step 3: Tokenize Text**

```python
num_words = 1000

en_tokenizer = Tokenizer(num_words=num_words)
en_tokenizer.fit_on_texts(english_sentences)
en_sequences = en_tokenizer.texts_to_sequences(english_sentences)

fr_tokenizer = Tokenizer(num_words=num_words, filters='')
fr_tokenizer.fit_on_texts(['<start> ' + s + ' <end>' for s in french_sentences])
fr_sequences = fr_tokenizer.texts_to_sequences(['<start> ' + s + ' <end>' for s in french_sentences])
```
✅ **Explanation:** We create tokenizers for English and French, adding `<start>` and `<end>` tokens on the French side.

---

### 🛠 **Step 4: Pad Sequences**

```python
max_en_len = max(len(seq) for seq in en_sequences)
max_fr_len = max(len(seq) for seq in fr_sequences)

encoder_input_data = pad_sequences(en_sequences, maxlen=max_en_len, padding='post')
decoder_input_data = pad_sequences([seq[:-1] for seq in fr_sequences], maxlen=max_fr_len - 1, padding='post')
decoder_target_data = pad_sequences([seq[1:] for seq in fr_sequences], maxlen=max_fr_len - 1, padding='post')
```
✅ **Explanation:** We pad English and French sequences to fixed lengths. For the decoder, we create two separate sets: input (shifted left) and target (shifted right).

---

### 🛠 **Step 5: One-Hot Encode Decoder Targets**

```python
num_decoder_tokens = len(fr_tokenizer.word_index) + 1
decoder_target_onehot = np.zeros((len(french_sentences), max_fr_len - 1, num_decoder_tokens), dtype='float32')
for i, seq in enumerate(decoder_target_data):
    for t, word_idx in enumerate(seq):
        if word_idx != 0:
            decoder_target_onehot[i, t, word_idx] = 1.0
```
✅ **Explanation:** We one-hot encode the decoder target data so we can use categorical cross-entropy loss.

---

### 🛠 **Step 6: Build Encoder-Decoder Model**

```python
embedding_dim = 64
latent_dim = 128

# Encoder
encoder_inputs = Input(shape=(max_en_len,))
enc_emb = Embedding(len(en_tokenizer.word_index) + 1, embedding_dim)(encoder_inputs)
encoder_lstm, state_h, state_c = LSTM(latent_dim, return_state=True)(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_fr_len - 1,))
dec_emb = Embedding(num_decoder_tokens, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
```
✅ **Explanation:** We set up the encoder (English) and decoder (French) LSTMs, linking them with the encoder’s final states.

---

### 🛠 **Step 7: Compile and Train the Model**

```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit([encoder_input_data, decoder_input_data], decoder_target_onehot, batch_size=2, epochs=100)
```
✅ **Explanation:** We compile the model with categorical cross-entropy loss and train it on the prepared input-target pairs.

---

### 🛠 **Step 8: Set Up Inference Models**

```python
# Encoder inference
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder inference
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb2 = Embedding(num_decoder_tokens, embedding_dim)(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)
```
✅ **Explanation:** We define separate encoder and decoder models for generating translations step by step during inference.

---

### 🛠 **Step 9: Define Translation Function**

```python
def translate(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.array([[fr_tokenizer.word_index['<start>']]])
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        for word, index in fr_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_fr_len:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
            target_seq = np.array([[sampled_token_index]])
            states_value = [h, c]
    return decoded_sentence
```
✅ **Explanation:** This function generates the French translation for a given English input by iteratively predicting one word at a time until the `<end>` token or maximum length is reached.
