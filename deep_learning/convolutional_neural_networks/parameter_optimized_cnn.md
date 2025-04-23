# 🎯 **Parameters Optimized During CNN Training**

CNNs learn **two main types of parameters**:  
1. **Weights**
2. **Biases**

These exist across different layers of the network:

---

## **1. Convolutional Layer Parameters**

- ✅ **Filter (Kernel) Weights**  
  - Each filter contains weights that define how it detects features.
  - Shape: `(filter height × filter width × input channels)`
  - If you have 32 filters of size `3×3×3`, you have `32 × (3×3×3) = 864` weights to learn.

- ✅ **Bias Terms**  
  - One bias per filter.
  - Allows flexibility in feature detection even with zero input.
  - For 32 filters → 32 bias terms.

- ✅ **Convolutional Layers Do not Share Parameters**
  - Each convolutional layer in a CNN has its own unique set of filters (weights and biases). The filters in one layer are completely independent of the filters in other layers.
  - For example, if you have three Convolutional Layers,
    - Layer 1 filters learn low-level features (edges, gradients, corners)
    - Layer 2 filters combine these into mid-level features (textures, shapes)
    - Layer 3 filters detect high-level features (eyes, wheels, faces)
      
- ✅ **What is Shared Then?**
  - Within the same layer, a filter’s weights are shared across spatial locations.
  - This is called parameter sharing: A filter slides across the input and uses the same weights at every position in that layer.

---

## **2. Fully Connected Layer Parameters**

- ✅ **Weights (Matrix)**  
  - Connect every neuron in one layer to every neuron in the next.
  - Shape: `(number of neurons in previous layer × number of neurons in current layer)`

- ✅ **Biases (Vector)**  
  - One bias for each neuron in the layer.
  - Allows shifting the activation function’s threshold.

---

## **3. Parameters in Optional Components**

- ✅ **Batch Normalization** (if used)
  - Learnable parameters:  
    - **γ (scale)** and **β (shift)** for each feature channel.
    - Helps the network adaptively normalize intermediate outputs.

- ✅ **Parametric Activation Functions** (like PReLU)
  - Learnable slope values for negative activations.
  - Adds flexibility to activation behavior.

---

# ✅ **Summary Table**

| **Layer Type**            | **Optimized Parameters**                            |
|---------------------------|-----------------------------------------------------|
| **Convolutional Layer**   | Filter Weights, Filter Biases                      |
| **Fully Connected Layer** | Weight Matrix, Bias Vector                         |
| **Batch Normalization**   | Scale (γ), Shift (β)                                |
| **PReLU / Learnable ReLU**| Negative slope values                              |

---

# 📌 What’s *Not* Learned (Fixed)

- **Stride**
- **Padding**
- **Filter size**
- **Pooling operations** (max/avg) — these are fixed algorithms, not learnable
