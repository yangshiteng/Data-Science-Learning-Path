## 🔁 **Backpropagation Through Time (BPTT)**

### 📚 What is BPTT?

**Backpropagation Through Time (BPTT)** is the extension of regular backpropagation used to train **RNNs**, which process **sequential data**.

Since RNNs share weights across **time steps**, we need to **unroll** the network through time and apply backpropagation to compute gradients at each step.

---

## 🔄 Why Can’t We Use Standard Backpropagation?

In feedforward networks, data flows layer by layer. In RNNs, the same operation is applied **repeatedly** over time using **shared parameters**.
So we can’t just apply standard backprop — we must account for **temporal dependencies** across time steps.

That’s what BPTT does!

---

## 🛠️ **Step-by-Step: How BPTT Works**

Let’s say we have an input sequence of length $T$:

$$
x_1, x_2, ..., x_T
$$

And at each time step, the RNN computes:

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
\quad ; \quad
y_t = W_{hy}h_t + b_y
$$

And we have a loss $\mathcal{L}_t$ at each time step.

---

### 🔹 Step 1: **Unroll the RNN**

We "unroll" the RNN into a deep network where each layer represents a time step:

```
x₁ → h₁ → y₁
      ↓
x₂ → h₂ → y₂
      ↓
...
xₜ → hₜ → yₜ
```

Each node uses **the same weights**.

---

### 🔹 Step 2: **Forward Pass**

For $t = 1$ to $T$:

* Compute $h_t$
* Compute $y_t$
* Compute loss $\mathcal{L}_t$

---

### 🔹 Step 3: **Compute Total Loss**

If we’re training with step-wise loss (e.g., language modeling):

$$
\mathcal{L}_{\text{total}} = \sum_{t=1}^{T} \mathcal{L}_t
$$

---

### 🔹 Step 4: **Backpropagation Through Time**

Now we backpropagate the error **from time $T$ to time $1$**, computing gradients with respect to the **shared parameters** $W_{xh}$, $W_{hh}$, and $W_{hy}$.

For each time step $t$, the gradient w\.r.t. hidden state depends on:

* Current loss $\mathcal{L}_t$
* Future gradients $\frac{\partial \mathcal{L}_{t+1}}{\partial h_t}$, $\frac{\partial \mathcal{L}_{t+2}}{\partial h_t}$, etc.

So gradients **accumulate backward through time**, which can cause:

* **Vanishing gradients** (→ gradients shrink too much)
* **Exploding gradients** (→ gradients grow too large)

---

### 🔹 Step 5: **Update Weights**

Once all gradients are computed, use an optimizer (e.g., SGD, Adam) to update weights:

$$
W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}
$$

---

## ⚠️ Issues with BPTT

| Problem                | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| 🚫 Vanishing Gradients | Gradients shrink over long sequences → can’t learn long-term memory |
| 💥 Exploding Gradients | Gradients grow uncontrollably → training becomes unstable           |
| 🐢 Slow Training       | RNNs process time steps sequentially → can't parallelize well       |

---

## ✂️ **Truncated BPTT**

To reduce computation and memory cost, we often use **Truncated BPTT**:

* Break the sequence into chunks (e.g., 20 steps)
* Unroll and backpropagate over only those time steps
* Keep the hidden state flowing between chunks

🧠 This makes training manageable, especially for long sequences.

---

## 🔁 Summary: BPTT Workflow

1. **Unroll** the RNN for $T$ steps
2. Perform a **forward pass**
3. Compute total **loss** over all time steps
4. Perform **backward pass** from $T \to 1$
5. Accumulate gradients **through time**
6. **Update weights**
