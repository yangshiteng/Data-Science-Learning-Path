# **Fully Connected Layer (Dense Layer)**

## **Definition**

A **Fully Connected Layer (FC Layer)**, also known as a **dense layer**, is a layer in which **each neuron is connected to every neuron** in the previous layer. It acts as a high-level reasoning layer that integrates the features extracted by earlier convolutional and pooling layers to perform tasks like **classification** or **regression**.

---

# **Position in CNN Architecture**

- Located **after the convolutional and pooling layers**.
- Takes the final, flattened feature maps and **interprets them to produce the final output**.
- Used especially in tasks like **image classification**, **object detection**, or **face recognition**.

---

## **Functionality**

Once the spatial feature maps have been reduced to a compact representation (often via pooling), they are:
1. **Flattened** into a 1D vector.
2. **Fed into one or more fully connected layers**.
3. The final FC layer outputs **logits** or **probabilities** (for classification tasks).

---

![image](https://github.com/user-attachments/assets/0953aaf3-14c2-40de-a6c1-23cd3a77f93e)

---

### **Common Activations Used**

- **ReLU**: For hidden layers
- **Softmax**: For the final output layer in classification (converts logits to probabilities)
- **Sigmoid**: In binary classification tasks

---

### **Example in Image Classification**

For a digit classification task (e.g., MNIST, 10 digits), the final FC layer might look like:
- Input size: `1 × 1024` (flattened from previous layers)
- FC Layer 1: 512 neurons → ReLU
- FC Layer 2: 128 neurons → ReLU
- **Output Layer**: 10 neurons → Softmax (one per digit class)

---

### **Key Roles of Fully Connected Layers**

| **Role**                       | **Explanation**                                       |
|-------------------------------|--------------------------------------------------------|
| **Combine features**          | Integrates all spatial features into a global view    |
| **Make decisions**            | Learns to associate feature patterns with output labels |
| **Final classification**      | Produces scores or probabilities for each class       |

---

### ✅ Summary

- **Fully Connected Layers** interpret and classify the features extracted earlier.
- They are **standard dense neural network layers** placed after the convolutional base.
- While powerful, they add many parameters, so they're often followed by **regularization** techniques like **dropout**.
