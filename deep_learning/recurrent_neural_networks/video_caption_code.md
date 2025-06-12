# 🧠🎥 TensorFlow/Keras Video Captioning Model using RNN

```python
# ----------------------------------------------
# 1. Imports
# ----------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed

# ----------------------------------------------
# 2. Simulated Preprocessed Data
# ----------------------------------------------

# Simulate 1 video represented by 10 frames, each with 2048 CNN features (like ResNet)
video_features = np.random.rand(1, 10, 2048)  # Shape: (batch_size, time_steps, feature_dim)

# Caption: "<BOS> a man is walking <EOS>" → [1, 5, 10, 15, 20, 2]
# Let's assume a vocab of 30 tokens
vocab_size = 30
caption_input = np.array([[1, 5, 10, 15, 20]])      # without <EOS>
caption_target = np.array([[5, 10, 15, 20, 2]])     # shifted left (what to predict next)

# Pad to fixed length (e.g., 10)
caption_input = tf.keras.preprocessing.sequence.pad_sequences(caption_input, maxlen=10, padding='post')
caption_target = tf.keras.preprocessing.sequence.pad_sequences(caption_target, maxlen=10, padding='post')

# Convert targets to one-hot for categorical crossentropy
caption_target = tf.keras.utils.to_categorical(caption_target, num_classes=vocab_size)

# ----------------------------------------------
# 3. Define the Model
# ----------------------------------------------

# Input 1: video features (sequence of frame features)
video_input = Input(shape=(10, 2048), name='video_input')

# Project video features to match LSTM input dimension
video_dense = TimeDistributed(Dense(256, activation='relu'))(video_input)

# Use an LSTM to encode video features into context vector
video_encoded, state_h, state_c = LSTM(256, return_state=True)(video_dense)

# Input 2: caption tokens (word IDs)
caption_input_seq = Input(shape=(10,), name='caption_input')
caption_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(caption_input_seq)

# Use another LSTM for decoding (caption generation)
decoder_lstm = LSTM(256, return_sequences=True)
decoder_output = decoder_lstm(caption_embedding, initial_state=[state_h, state_c])

# Output layer: predicts vocab distribution at each time step
output = Dense(vocab_size, activation='softmax')(decoder_output)

# Build the model
model = Model(inputs=[video_input, caption_input_seq], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Show model summary
model.summary()
```

---

## 🏋️ Training

```python
# Train on simulated data
model.fit([video_features, caption_input], caption_target, epochs=5, batch_size=1)
```

---

## 🧪 Inference (Caption Generation)

```python
# To generate captions, use a loop:
# 1. Feed video features
# 2. Start with <BOS> token
# 3. At each time step, predict next word
# 4. Feed predicted word back in
# 5. Stop when <EOS> token is predicted

# For simplicity, we omit beam search or greedy sampling here.
```

---

## 📌 Notes

- `video_input`: 10 frames, each with 2048-dim feature vectors (from a pretrained CNN like ResNet).
- `caption_input`: sequence of word indices, padded to fixed length.
- `caption_target`: one-hot encoded version of the expected output.
- `Embedding + LSTM`: generates the caption step by step.
- Model is trained using `categorical_crossentropy` loss over vocabulary predictions.

---

## 🧠 Extensions You Can Try

- Use real CNN features from a pretrained model (e.g., ResNet50 from `tf.keras.applications`)
- Add attention mechanism over video frames
- Replace RNN with Transformer decoder
- Use beam search during inference for better captions
