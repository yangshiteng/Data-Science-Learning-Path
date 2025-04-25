## 📱 **MobileNet: Efficient CNN for Mobile and Edge Devices**

### 📌 **Overview**

| Feature               | Description                                         |
|------------------------|-----------------------------------------------------|
| **Name**               | MobileNet                                           |
| **Authors**            | Andrew G. Howard et al. (Google)                    |
| **Published**          | 2017 (original MobileNet v1)                        |
| **Main Goal**          | Create fast, low-latency CNNs for mobile and embedded use |
| **Key Innovation**     | **Depthwise Separable Convolutions**               |
| **Used In**            | Object detection, face recognition, image classification (e.g., in TensorFlow Lite) |

---

## 🧠 **Key Design Innovation: Depthwise Separable Convolution**

MobileNet reduces computational cost by **factorizing a standard convolution** into two separate layers:

### 🔧 Traditional Convolution (costly):
- Applies filters over **all input channels** at once
- Very expensive when input has many channels

### ✅ **Depthwise Separable Convolution**:
1. **Depthwise Convolution**:
   - Apply **one filter per input channel**
   - No mixing between channels yet

2. **Pointwise Convolution**:
   - 1×1 convolution to **combine features across channels**

> This reduces computation by **8–9×**, without a large drop in accuracy.

---

## 🧱 **MobileNet v1 Architecture (Simplified)**

| **Layer Type**        | **Details**                           | **Output Shape (example input 224×224×3)** |
|------------------------|----------------------------------------|---------------------------------------------|
| Conv2D                 | 3×3, stride 2                          | 112×112×32                                  |
| Depthwise + Pointwise | 3×3 DW, 1×1 PW, stride 1               | 112×112×64                                  |
| Depthwise + Pointwise | 3×3 DW, 1×1 PW, stride 2               | 56×56×128                                   |
| ...                   | Repeats with increasing channels       | ↓                                           |
| Depthwise + Pointwise | Final block: 1024 channels             | 7×7×1024                                    |
| Global Avg Pool       |                                        | 1×1×1024                                    |
| Fully Connected        | 1000-way softmax (ImageNet classes)   | 1000                                        |

---

## ⚙️ **Model Customization with Width and Resolution Multipliers**

### ✅ **Width Multiplier (α)**:
- Reduces the number of channels
- E.g., α = 0.5 cuts channels in half → lighter model

### ✅ **Resolution Multiplier (ρ)**:
- Reduces input image size (e.g., 224 → 160 or 128)
- Less computation for smaller resolutions

This makes MobileNet highly **scalable for different devices** and resource constraints.

---

## 📈 **Performance (MobileNet v1)**

| Metric                    | Value                         |
|---------------------------|-------------------------------|
| **Top-1 Accuracy (ImageNet)** | ~70–72% (with α = 1.0)     |
| **Top-5 Accuracy**        | ~89.5%                        |
| **Parameters**            | ~4.2 million                  |
| **FLOPs**                 | ~575 million                  |
| **Size**                  | ~16MB                         |

---

## 🚀 **MobileNet Versions**

| Version        | Highlights                                                |
|----------------|-----------------------------------------------------------|
| **MobileNet v1** | Depthwise separable convolutions                        |
| **MobileNet v2** | Inverted residuals + linear bottlenecks                 |
| **MobileNet v3** | Neural architecture search (NAS) + SE blocks + swish activation |
| **MobileNetV3-Lite** | Optimized for edge devices with minimal compute      |

---

## ✅ **Summary Table**

| **Aspect**             | **MobileNet v1**                  |
|------------------------|------------------------------------|
| Year                   | 2017                               |
| Parameters             | ~4.2 million                       |
| Main Innovation        | Depthwise separable convolutions   |
| Ideal Use              | Mobile apps, embedded systems      |
| Speed vs. Accuracy     | Excellent trade-off                |
| Customizable           | Via α (width) and ρ (resolution)   |
