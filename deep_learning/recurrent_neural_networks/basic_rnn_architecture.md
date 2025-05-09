## 🎯 **Basic RNN Architecture**

![image](https://github.com/user-attachments/assets/dda805ad-b754-4369-8788-ff8dea98dc66)

### 🧠 **1. Overview**

The **Basic Recurrent Neural Network (RNN)** is designed to model **sequential data**. Unlike feedforward networks, it contains **loops** that allow information to persist over time—acting like a **memory**. This makes it ideal for tasks like:

* 📝 Text generation
* 🔊 Speech recognition
* 📈 Time series forecasting

---

### 🔧 **2. Core Components**

At each time step $t$, the RNN consists of:

* 📥 **Input vector $x_t$**: The current element in the input sequence
* 🔁 **Hidden state $h_t$**: Memory that carries information from previous time steps
* 📤 **Output $y_t$**: The output or prediction at time step $t$

### 🔢 Formulas:

![image](https://github.com/user-attachments/assets/1321a6e3-10e8-42b0-8cc3-cee14c2516dd)

---

### 🔁 **3. Recurrent Loop**

The key feature is the **looping structure**:
⏩ The hidden state $h_t$ at time $t$ becomes part of the input for time $t+1$.

This creates a **temporal dependency**, allowing the network to "remember" past information and use it to inform future outputs.

---

### 📚 **4. Sequence Processing**

To process a sequence $x_1, x_2, ..., x_T$:

1. 🛑 Initialize $h_0 = 0$
2. 🔄 For each time step $t \in \{1, ..., T\}$:

   * Update hidden state $h_t$
   * Compute output $y_t$

This process is known as **unrolling the RNN over time**.

---

### ⚠️ **5. Limitations**

Despite its simplicity, basic RNNs have some drawbacks:

* 🚫 **Short-Term Memory**: Struggles with long-range dependencies
* 📉 **Vanishing Gradients**: Gradients shrink as they backpropagate through time
* 📈 **Exploding Gradients**: Gradients can also grow too large, destabilizing training

🧪 These issues motivated the development of more advanced architectures like:

* 🔒 **LSTM (Long Short-Term Memory)**
* ⚙️ **GRU (Gated Recurrent Unit)**
