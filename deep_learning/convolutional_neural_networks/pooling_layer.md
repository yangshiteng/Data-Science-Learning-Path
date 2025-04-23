# **Pooling Layer**

## 🔹 **What is Pooling?**

**Pooling** is a downsampling operation used to **reduce the spatial dimensions** (height and width) of the feature maps while keeping the **most important information**. It helps make the network more efficient and less sensitive to slight shifts or distortions in the input image.

> 💡 Think of pooling as summarizing a region of the image — keeping only the most important detail.

---

## 🔹 **Why Use Pooling?**

- Reduces the number of parameters and computations.
- Controls **overfitting**.
- Makes feature representations **invariant to small translations or noise** (e.g., moving an object slightly in the image doesn’t change the result much).

---

## 🔹 **Types of Pooling**

1. **Max Pooling** (most common)
   - Takes the **maximum** value in each region.
   - Keeps the **strongest feature**.
   - Example: From a 2×2 region `[1, 3, 2, 8]`, max pooling returns `8`.

2. **Average Pooling**
   - Takes the **average** of the values in each region.
   - Example: `[1, 3, 2, 8]` → `(1+3+2+8)/4 = 3.5`

---

## 🔹 **How It Works**

1. Choose a **pooling window size** (e.g., `2×2`) and **stride** (usually 2).
2. Slide the window over the feature map.
3. Apply the pooling operation (max or average) in each region.
4. Output a **smaller feature map**.

![image](https://github.com/user-attachments/assets/b16a41ff-60a2-49ff-ac43-296551aa9363)

---

## 🧠 Example

Let’s say your feature map is `28×28` after activation. If you apply:
- **Max Pooling**
- Window size: `2×2`
- Stride: `2`

Then your output becomes:  
➡️ `14 × 14` feature map (with reduced detail but key info kept)

---

# 📉 Summary Table

| Pooling Type   | Output | Purpose                            |
|----------------|--------|------------------------------------|
| **Max Pooling**| Max    | Keeps strongest feature (most used)|
| **Avg Pooling**| Mean   | Smooths out data                   |

---
