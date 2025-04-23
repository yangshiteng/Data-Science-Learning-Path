# **Bias in Convolutional Neural Networks (CNNs)**

## **Definition**

In CNNs, a **bias** is a trainable scalar value added to the output of a convolution operation **before** the activation function is applied. It allows the network to **shift the activation** and adds flexibility to the learning process, even when the input values are zero.

---

![image](https://github.com/user-attachments/assets/00c83a63-5982-49f3-bd6f-00553bb70240)

---

## **Purpose of Bias**

- **Shift Activation**: Bias enables the neuron (or filter) to activate even when the input is zero or near zero.
- **Control Threshold**: It effectively acts as a threshold the neuron must exceed to fire.
- **Learnable Parameter**: Like weights, bias is learned during training via backpropagation and gradient descent.
- **Prevents Dead Neurons**: Especially useful with ReLU, which zeros out negative values — a bias can shift values to stay active.

---

## **Bias in CNN Layers**

- Each filter in a convolutional layer typically has **one associated bias value**.
- So, if a convolutional layer has **N filters**, it will have **N bias terms**.
- These biases are added to **every position** in the corresponding feature map (they are broadcasted across spatial dimensions).

---

![image](https://github.com/user-attachments/assets/29ecbee8-fc83-4529-8354-c784d96fd665)

---

### ✅ Summary

| **Aspect**       | **Bias in CNNs**                          |
|------------------|-------------------------------------------|
| Type             | Scalar (one per filter)                   |
| When used        | After convolution, before activation      |
| Purpose          | Shifts the activation, increases flexibility |
| Learnable?       | Yes                                       |
| Benefit          | Helps with training stability and accuracy|
