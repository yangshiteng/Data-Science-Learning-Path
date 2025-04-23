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

![image](https://github.com/user-attachments/assets/424c24db-a9e9-4b33-9f99-80156ade0196)

![image](https://github.com/user-attachments/assets/f58ba924-4ff0-455f-a51e-f8b36a55e565)
![image](https://github.com/user-attachments/assets/6f77c45e-cadf-4fa1-929f-63cefe01c12e)

![image](https://github.com/user-attachments/assets/959be97f-4c03-4e27-8470-8a7ffd92c638)

![image](https://github.com/user-attachments/assets/cb22e54c-f6ea-49b8-a276-64f0404397c0)

![image](https://github.com/user-attachments/assets/850cb0ad-786e-42c3-b62a-48846ea774f8)

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
