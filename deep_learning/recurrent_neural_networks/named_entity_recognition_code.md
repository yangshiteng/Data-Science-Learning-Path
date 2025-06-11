# Named Entity Recognition (NER) using BiLSTM in TensorFlow/Keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# -----------------------------
# 1. Dummy Preprocessed Data (Replace with real data loading)
# -----------------------------
# Example word index and label index
word2idx = {"PAD": 0, "Barack": 1, "Obama": 2, "visited": 3, "Paris": 4, ".": 5}
label2idx = {"PAD": 0, "B-PER": 1, "I-PER": 2, "O": 3, "B-LOC": 4}
idx2label = {i: l for l, i in label2idx.items()}

# Example sentence (token IDs) and labels (label IDs)
X = [[1, 2, 3, 4, 5]]  # "Barack Obama visited Paris ."
y = [[1, 2, 3, 4, 3]]  # "B-PER I-PER O B-LOC O"

# Padding
max_len = 10
X = pad_sequences(X, maxlen=max_len, padding='post')
y = pad_sequences(y, maxlen=max_len, padding='post')
y = [to_categorical(i, num_classes=len(label2idx)) for i in y]  # One-hot encoding

# -----------------------------
# 2. Define Model
# -----------------------------
input = Input(shape=(max_len,))
model = Embedding(input_dim=len(word2idx), output_dim=64, input_length=max_len, mask_zero=True)(input)
model = Bidirectional(LSTM(units=64, return_sequences=True))(model)
model = TimeDistributed(Dense(len(label2idx), activation="softmax"))(model)

model = Model(input, model)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------------------
# 3. Train Model
# -----------------------------
X = np.array(X)
y = np.array(y)
model.fit(X, y, batch_size=2, epochs=5)

# -----------------------------
# 4. Inference Example
# -----------------------------
preds = model.predict(X)
preds_idx = np.argmax(preds[0], axis=-1)
pred_labels = [idx2label[i] for i in preds_idx]
print("Predicted Labels:", pred_labels)
