# **Feature Map (Activation Map)**

## 🔹 **What is a Feature Map?**

A **feature map** (also called an **activation map**) is the output produced by applying a filter to the input image or to the previous layer’s output. It represents **where and how strongly** a certain pattern or feature appears in different parts of the image.

> 💡 Imagine you apply an edge-detecting filter — the resulting feature map will highlight the parts of the image that contain those edges.

---

## 🔹 **How It's Created**

1. A **filter (kernel)** is slid over the input using a certain **stride**.
2. At each position, **element-wise multiplication and summation** are done between the filter and a patch of the input.
3. The resulting value is placed in the feature map.
4. This process continues until the entire input is covered.

Each **filter produces one feature map**.

---

## 🔹 **Shape of a Feature Map**

If:
- Input size = `H × W × C`
- Filter size = `f × f`
- Padding = `P`
- Stride = `S`
- Number of filters = `K`

Then the **output (feature map)** size is:
```
Output height = (H - f + 2P) / S + 1  
Output width  = (W - f + 2P) / S + 1  
Output depth  = K  (one feature map per filter)
```

---

## 🔹 **Why Feature Maps Matter**

- They **highlight the presence and location** of specific features in the image.
- Early layers produce feature maps for simple features like **edges** and **textures**.
- Deeper layers produce maps for complex patterns like **eyes**, **faces**, or **objects**.
- These maps are passed along the network and help in **decision-making** at the output (e.g., classifying an image as a "cat").

---

## 🧠 **Example**

Let’s say you input a `28×28×1` grayscale image and use:
- 5 filters of size `3×3`
- Padding = 1
- Stride = 1

Then:
- Output feature maps will be: `28×28×5`

That means: 5 different versions of the input, each highlighting a different learned feature.

---

## 📊 Visual Intuition

- Bright spots = Strong response to the filter (i.e., feature is present)
- Dark spots = Weak or no response (feature is absent)

---

Would you like a visual or code example of generating a feature map from an image using a simple filter? Or should we continue to the **Activation Function (ReLU)** next?
