# 📚 **CNN Application with TensorFlow & Keras**

---

# 🚀 **We Will Build:**
- A CNN to **classify images** from the **CIFAR-10 dataset** (10 classes: airplanes, cars, cats, dogs, etc.).
- Using **TensorFlow 2.x** and **Keras Sequential API**.

---

# 🛠️ **Step 1: Install TensorFlow**

If you haven't installed TensorFlow yet:

```bash
pip install tensorflow
```

---

# 🛠️ **Step 2: Import Libraries**

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
```

✅ These imports give you **full access** to TensorFlow, Keras Layers, and Matplotlib for visualization.

---

# 🛠️ **Step 3: Load and Prepare the CIFAR-10 Dataset**

```python
# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0,1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Check shapes
print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)
```

✅ Normalization makes training more stable (pixels go from 0–255 → 0–1).

---

# 🛠️ **Step 4: Visualize Sample Data**

(Optional but good practice!)

```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(class_names[int(y_train[i])])
plt.show()
```

✅ Helps you **understand** what the data looks like.

---

# 🛠️ **Step 5: Build the CNN Model**

```python
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten and Fully Connected Layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes
])
```

✅ This CNN has:
- **3 convolutional blocks** (Conv2D + ReLU + MaxPooling)
- **Flatten** the 2D maps into 1D
- **2 dense layers** (one hidden, one output)

---

# 🛠️ **Step 6: Compile the Model**

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

✅ **Adam optimizer** = good default choice.  
✅ **Sparse categorical crossentropy** = used because labels are integers, not one-hot encoded.

---

# 🛠️ **Step 7: Train the Model**

```python
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))
```

✅ This trains the model for **10 epochs**, and automatically checks performance on the **test set** after every epoch.

---

# 🛠️ **Step 8: Evaluate the Model**

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')
```

✅ After training, evaluate how well the model generalizes to unseen data.

---

# 🛠️ **Step 9: Plot Training and Validation Accuracy**

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('Training vs Validation Accuracy')
plt.show()
```

✅ Always visualize training curves to spot **overfitting** or **underfitting**!

---

# 📈 **Summary of Full Steps**

| Step                  | Purpose                        |
|------------------------|--------------------------------|
| Install TensorFlow     | Setup deep learning environment |
| Import libraries       | Load TensorFlow and helpers    |
| Load and normalize data | Prepare training and testing data |
| Visualize sample images | Understand dataset visually   |
| Build CNN model        | Stack Conv and Dense layers    |
| Compile model          | Set optimizer and loss         |
| Train model            | Fit model to training data     |
| Evaluate model         | Check performance on test data |
| Plot results           | Understand training behavior   |

---

# 🧠 **Final Takeaway**

> **TensorFlow + Keras** allows you to build, train, and evaluate a **strong CNN model**  
> for real-world image classification tasks in **just a few dozen lines of code** —  
> making deep learning accessible, powerful, and production-ready!
