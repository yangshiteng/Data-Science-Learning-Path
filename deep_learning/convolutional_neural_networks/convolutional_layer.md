# **Convolutional Layer**

## **What It Does**

The **convolutional layer** applies a set of **filters** (also called kernels) to the input image to produce **feature maps**. These feature maps highlight specific patterns like edges, corners, textures, or more abstract concepts as we go deeper.

> 🧠 Think of it like sliding a small window across the image and checking for specific patterns.

---

## **How It Works**

- Each **filter** is a small matrix (e.g., 3×3 or 5×5).
- The filter **slides** (or convolves) across the image with a certain **stride**.
- At each position, it performs **element-wise multiplication** and **sums the results** — this produces one number for that spot in the output feature map.
- The output of applying one filter to the image is called a **feature map** or **activation map**.

---

## **Mathematically**

If:
- Input image shape = `(H, W, C)`
- Filter size = `(f, f)`
- Number of filters = `K`
- Stride = `S`
- Padding = `P`

Then:
- Output feature map size =  
  ```
  ((H - f + 2P)/S + 1) × ((W - f + 2P)/S + 1) × K
  ```

---

## **Why It Matters**

- The convolutional layer **learns local patterns** like edges and textures.
- Filters are **learned automatically** during training.
- Early layers learn **low-level features** (edges, gradients), while deeper layers capture **high-level features** (object parts, whole shapes).

---

## **Example**

If you input a `32×32×3` color image and use:
- 16 filters of size `3×3`
- Stride = 1
- Padding = 1

Then the output will be a feature map of shape:  
`32 × 32 × 16`

---

## **Next Step After Convolutional Layer?**
Usually, it's followed by:
1. An **activation function** (commonly **ReLU**) to introduce non-linearity
2. Then often a **Pooling layer** to reduce spatial size
