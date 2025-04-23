# **Stride and Padding**

---

## 🧱 **Stride**

### 🔹 What is Stride?

**Stride** is the number of pixels the filter moves (or “slides”) across the input image at each step.

- **Stride = 1**: The filter moves one pixel at a time — very detailed scan.
- **Stride = 2**: The filter skips every other pixel — faster, but less detail.

### 🔹 How Stride Affects Output Size

- **Higher stride = smaller output feature map** (less spatial resolution).
- Think of it like zooming out or downsampling.

> 📐 Example:
If you have a `5×5` input and use a `3×3` filter:
- **Stride 1** → Output size = `3×3`
- **Stride 2** → Output size = `2×2`

![image](https://github.com/user-attachments/assets/fe3f70c4-e019-4488-8d92-4d444de1cc7c)

---

## 🧱 **Padding**

### 🔹 What is Padding?

**Padding** refers to adding extra pixels (usually zeros) around the border of the input image **before** applying the filter. This is done so that:
- The output feature map doesn’t shrink too much
- The filter can properly scan the edges of the image

### 🔹 Types of Padding

1. **Valid Padding (No Padding)**  
   - No extra pixels added  
   - Output is **smaller** than input  
   - Sometimes called "valid" because the filter only slides over valid positions

2. **Same Padding (Zero Padding)**  
   - Add zeros around the image  
   - Output size stays **the same** as input (assuming stride = 1)  
   - Helps preserve spatial dimensions

### 🔹 Why Padding is Important

- Without padding, images shrink with every convolution layer — you’d quickly lose useful information.
- With padding, you can control and maintain feature map size deeper into the network.

![image](https://github.com/user-attachments/assets/fc85ecae-86b8-4ca6-ac94-6527b0fe9a71)

---

# 🧠 Summary Table

| **Setting** | **Effect** |
|-------------|------------|
| **Stride** | Controls how much the filter moves; larger stride = faster but less detail |
| **Padding** | Controls whether output size shrinks or stays the same; prevents losing edge information |
