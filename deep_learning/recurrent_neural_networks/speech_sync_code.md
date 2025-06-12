## 🧠 RNN-Based Speech Synthesis: Tacotron-like Acoustic Model (Text → Spectrogram)

```python
# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

---

### 1️⃣ Simulated Dataset

```python
# Define a simple vocabulary
char2idx = {"PAD": 0, "H": 1, "E": 2, "L": 3, "O": 4}
idx2char = {i: c for c, i in char2idx.items()}

# Input: "HELLO" → [1, 2, 3, 3, 4]
X = [[1, 2, 3, 3, 4]]
X = pad_sequences(X, maxlen=10, padding='post')

# Simulated output: 20 frames of 80-dim mel spectrogram
y = np.random.rand(1, 20, 80)
```

---

### 2️⃣ Define the Encoder-Decoder Model

```python
# Input sequence of character IDs
input_seq = Input(shape=(10,), name='text_input')
embedded = Embedding(input_dim=len(char2idx), output_dim=64, mask_zero=True)(input_seq)

# Encoder
encoder_outputs, state_h, state_c = LSTM(128, return_state=True, return_sequences=True)(embedded)

# Decoder input placeholder
decoder_input = Input(shape=(None, 128), name='decoder_input')

# Decoder
decoder_lstm = LSTM(128, return_sequences=True)
decoder_outputs = decoder_lstm(decoder_input, initial_state=[state_h, state_c])

# Project to mel spectrogram
mel_output = Dense(80, activation='linear', name='mel_output')(decoder_outputs)

# Model
model = Model(inputs=[input_seq, decoder_input], outputs=mel_output)
model.compile(optimizer='adam', loss='mse')
model.summary()
```

---

### 3️⃣ Training with Dummy Decoder Input

```python
# Create fake decoder input by repeating encoder output (simplified for demo)
decoder_input_data = np.repeat(np.expand_dims(encoder_outputs.numpy(), axis=1), 20, axis=1)

# Train
model.fit([X, decoder_input_data], y, epochs=5, batch_size=1)
```

---

### 4️⃣ Predict Mel-Spectrogram

```python
predicted_mel = model.predict([X, decoder_input_data])
print("Predicted mel spectrogram shape:", predicted_mel.shape)  # (1, 20, 80)
```

---

### ✅ Notes

- This simulates text-to-spectrogram prediction using RNNs.
- Real Tacotron models include attention mechanisms and stop tokens.
- Final audio is produced by feeding the spectrogram into a vocoder like Griffin-Lim or HiFi-GAN.

