## ⚙️ **Weight Initialization Strategies**

### 🧠 **Why Is Weight Initialization Important?**

When training a neural network, how you **initialize the weights** has a huge impact on:

* 🔄 Convergence speed
* 📉 Whether gradients vanish or explode
* ✅ Overall model performance

Poor initialization can lead to:

* Very slow learning
* Getting stuck in bad local minima
* Instability during training

---

## 🎯 **Goals of Good Initialization**

* Prevent **activations from shrinking or blowing up** layer by layer
* Keep **gradient flow stable** during backpropagation
* Maintain **symmetry breaking** (so different neurons learn different things)

---

## 🔧 **Common Weight Initialization Methods**

### 1. ✳️ **Zero Initialization (Don’t Do This!)**

* All weights are set to 0
* ❌ Problem: All neurons behave the same — no learning happens
* ✅ OK only for **biases**, not weights

---

### 2. 🔀 **Random Initialization**

* Small random numbers (e.g., Gaussian or uniform)
* Better than zeros, but still not ideal — doesn’t account for layer size
* Used mostly in early networks

---

### 3. 🟩 **Xavier Initialization (Glorot Initialization)**

* Designed for **tanh** or **sigmoid** activations
* Keeps variance of activations and gradients stable across layers

For uniform distribution:

$$
W \sim U\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)
$$

✅ Used in: **shallow networks**, **autoencoders**, **basic RNNs**

---

### 4. 🔷 **He Initialization (Kaiming Initialization)**

* Designed for **ReLU** and **Leaky ReLU** activations
* Helps prevent activations from dying (outputting zero)

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
$$

✅ Used in: **CNNs**, **deep feedforward**, **ResNets**

---

### 5. 🔁 **Orthogonal Initialization**

* Used often in **RNNs**, especially for the recurrent weight matrix $W_{hh}$
* Helps preserve **gradient magnitude over time steps**
* Makes backpropagation through time more stable

✅ Particularly effective for long-sequence RNNs and LSTMs

---

### 6. 🔄 **Identity Initialization (for RNNs)**

* Initialize $W_{hh}$ as an identity matrix
* Encourages information to **pass unchanged** across time steps early in training
* Works well with **ReLU-based RNNs**

---

## 🤖 **Weight Initialization for RNNs (Recap)**

| Component                  | Recommended Strategy                            |
| -------------------------- | ----------------------------------------------- |
| Input weights $W_{xh}$     | Xavier or He Initialization                     |
| Recurrent weights $W_{hh}$ | Orthogonal or Identity Initialization           |
| Biases                     | Zeros (or small positive for LSTM forget gates) |

---

## 🧾 Summary

| Strategy         | Best For                 | Helps With                       |
| ---------------- | ------------------------ | -------------------------------- |
| Xavier           | tanh/sigmoid activations | Balanced gradients/activations   |
| He               | ReLU activations         | Avoiding dead neurons            |
| Orthogonal       | RNNs, long sequences     | Stable time-step gradient flow   |
| Identity         | ReLU-RNNs                | Preserving information over time |
| Zero (bias only) | Bias terms               | Safe for initializing biases     |
