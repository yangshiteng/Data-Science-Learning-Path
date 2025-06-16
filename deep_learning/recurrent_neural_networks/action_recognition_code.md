```python
# 📦 Step 1: Imports and Setup
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# 🧪 Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 🧰 Step 2: Simulate Feature Data and Labels
# Simulate 100 video samples
# Each video: 16 frames → each frame has 2048 CNN features (e.g., from ResNet)
X_train = np.random.rand(100, 16, 2048)  # shape: (samples, time_steps, feature_dim)

# Simulate action labels (e.g., 5 action classes)
y_labels = np.random.randint(0, 5, 100)  # 5 action classes
Y_train = to_categorical(y_labels, num_classes=5)

# 🧠 Step 3: Define the RNN Model for Action Recognition
# Input shape: sequence of 2048-dim CNN features
input_layer = Input(shape=(16, 2048), name="video_feature_input")

# LSTM processes temporal sequence of frame features
x = LSTM(256, return_sequences=False)(input_layer)  # Outputs final hidden state

# Fully connected layer to map to class scores
output = Dense(5, activation='softmax', name="action_class_output")(x)

# Build the model
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 🏋️ Step 4: Train the Model
model.fit(X_train, Y_train, batch_size=8, epochs=10, validation_split=0.1)

# 🔍 Step 5: Evaluate or Predict
sample_input = X_train[:1]  # Pick one video sample
prediction = model.predict(sample_input)
print("Predicted class probabilities:", prediction)
print("Predicted class index:", np.argmax(prediction))
```
