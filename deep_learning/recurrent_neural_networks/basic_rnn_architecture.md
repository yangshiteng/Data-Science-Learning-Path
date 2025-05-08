## 🎯 **Basic RNN Architecture**

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

🔢 **Formulas**:

$$
h_t = \tanh(\underbrace{W_{xh}x_t}_{\text{input}} + \underbrace{W_{hh}h_{t-1}}_{\text{memory}} + b_h)
$$

$$
y_t = W_{hy}h_t + b_y
$$

🧩 **Where**:

* $W_{xh}$: Weights from input to hidden
* $W_{hh}$: Recurrent weights (memory)
* $W_{hy}$: Weights from hidden to output
* $\tanh$: Activation function to keep values bounded

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
