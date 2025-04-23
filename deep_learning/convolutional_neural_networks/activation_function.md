# **Activation Function (ReLU)**

## 🔹 **What is an Activation Function?**

An **activation function** is a mathematical operation applied **element-wise** to the values in a **feature map**. Its main job is to introduce **non-linearity** into the CNN, allowing the network to learn **complex patterns** beyond just simple linear relationships.

> 💡 Without activation functions, the entire CNN would behave like a linear model — no matter how deep it is!

---

## 🔹 **Most Common Activation in CNNs: ReLU**

**ReLU (Rectified Linear Unit)** is the default activation function in most CNNs.

**Formula**:
```
ReLU(x) = max(0, x)
```

**How it works**:
- If the value is **positive**, it stays the same.
- If the value is **negative**, it becomes 0.

So it turns all the **negative values** in the feature map to **zero** and keeps the **positive ones**.

---

## 🔹 **What ReLU Looks Like**

| Input Value | ReLU Output |
|-------------|-------------|
| -5          | 0           |
| 0           | 0           |
| 3           | 3           |
| 7.2         | 7.2         |

---

## 🔹 **Why ReLU is Important**

- **Non-linearity**: Allows the network to model more complex functions.
- **Efficient**: Very simple computation — fast to apply.
- **Prevents saturation**: Unlike sigmoid/tanh, ReLU doesn’t suffer from vanishing gradients as much.

---

## 🔹 Variants of ReLU

- **Leaky ReLU**: Allows a small negative slope instead of turning all negatives to zero.
- **Parametric ReLU (PReLU)**: Learns the slope of the negative part.
- **ELU / SELU**: Other smooth alternatives used in some cases.

---

## 🧠 Summary

After a convolution operation produces a feature map:
1. The activation function is applied to **each value** in the feature map.
2. Negative values are usually set to **0** (if using ReLU).
3. The result is a **non-linear feature map** that goes into the next layer.

---
