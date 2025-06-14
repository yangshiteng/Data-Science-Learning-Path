# 🎥🔮 Video Frame Prediction using ConvLSTM (TensorFlow/Keras)

This notebook demonstrates how to implement a simple RNN-based video frame predictor using ConvLSTM layers in TensorFlow/Keras.

---

## 📦 Step 1: Imports and Setup

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
```

---

## 🧰 Step 2: Parameters and Simulated Dataset

```python
# Frame and image configuration
frames = 10         # Number of frames per sequence
height = 64         # Frame height
width = 64          # Frame width
channels = 1        # Grayscale images

# Simulate training data
# Input: sequences of 10 frames
# Output: predicted future sequences of 10 frames
X_train = np.random.rand(100, frames, height, width, channels)
Y_train = np.random.rand(100, frames, height, width, channels)
```

---

## 🧠 Step 3: Define the ConvLSTM Model

```python
# Input layer: expects a sequence of frames
input_layer = Input(shape=(frames, height, width, channels))

# First ConvLSTM layer: captures spatiotemporal features
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(input_layer)
x = BatchNormalization()(x)

# Second ConvLSTM layer: refines temporal features
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
x = BatchNormalization()(x)

# Output layer: uses 3D convolution to produce predicted frame sequence
output = Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(x)

# Compile model
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')
model.summary()
```

---

## 🏋️ Step 4: Train the Model

```python
# Train with batch size 4 for 10 epochs
model.fit(X_train, Y_train, batch_size=4, epochs=10, validation_split=0.1)
```

---

## 🔍 Step 5: Predict Future Frames

```python
# Select one sample and predict its future frames
sample_input = X_train[:1]
predicted_output = model.predict(sample_input)

# Check output shape
print("Predicted output shape:", predicted_output.shape)
# Expected: (1, 10, 64, 64, 1)
```

---

## ✅ Notes

- The model uses ConvLSTM2D to learn both spatial (pixel structure) and temporal (motion) information.
- Output is generated frame-by-frame, using previous frames as context.
- You can replace simulated data with real sequences from datasets like Moving MNIST, UCF-101, or KTH.