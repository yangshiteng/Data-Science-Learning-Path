# 📦 Step 1: Imports and Setup
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D

# 🧪 For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 🧰 Parameters
frames = 10         # Number of frames in input/output
height = 64         # Frame height
width = 64          # Frame width
channels = 1        # Grayscale

# 🧾 Simulated dataset
X_train = np.random.rand(100, frames, height, width, channels)  # Input sequence
Y_train = np.random.rand(100, frames, height, width, channels)  # Target sequence

# 🧠 Step 2: Define ConvLSTM Model
input_layer = Input(shape=(frames, height, width, channels))

# ConvLSTM layer
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(input_layer)
x = BatchNormalization()(x)

# Another ConvLSTM layer (optional)
x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding="same", return_sequences=True, activation="relu")(x)
x = BatchNormalization()(x)

# Final 3D convolution to produce output frames
output = Conv3D(filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same")(x)

# Build model
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')
model.summary()

# 🏋️ Step 3: Train the Model
model.fit(X_train, Y_train, batch_size=4, epochs=10, validation_split=0.1)

# 🔍 Step 4: Evaluate / Predict
sample_input = X_train[:1]  # Pick one sample
predicted_output = model.predict(sample_input)
print("Predicted output shape:", predicted_output.shape)  # Expected: (1, 10, 64, 64, 1)
