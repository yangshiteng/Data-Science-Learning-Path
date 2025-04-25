## 🧠 **ResNet (Residual Network)**

### 📌 **Overview**

| Feature            | Description                                           |
|--------------------|-------------------------------------------------------|
| **Name**           | ResNet (Residual Neural Network)                     |
| **Authors**        | Kaiming He et al. (Microsoft Research)               |
| **Published**      | 2015 (CVPR, *“Deep Residual Learning for Image Recognition”*) |
| **Key Innovation** | **Residual connections** (aka **skip connections**)   |
| **Top Achievement**| **Won ILSVRC 2015** (ImageNet) with 3.57% top-5 error |
| **Depths Tested**  | ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152 |

---

## 🔍 **Motivation: Why Residuals?**

As CNNs get deeper, they should theoretically perform better — but in practice:
- Deeper models **start to perform worse** due to **vanishing gradients**.
- Optimization becomes harder.
- Networks may **degrade** even when more layers are added.

### ✅ **ResNet’s Solution: Residual Learning**

![image](https://github.com/user-attachments/assets/a37c366b-436c-422e-a38b-f3054bd9056a)

---

## 🧱 **Basic ResNet Block (Residual Block)**

### 🔧 **Structure of a Residual Block:**

```text
Input → [Conv → BN → ReLU → Conv → BN] → Add(Input) → ReLU
```

- The **input is added** back to the output of the stacked conv layers.
- This “shortcut” helps **information and gradients flow more easily**.

---

## 🏗️ **ResNet Variants**

| **Model**     | **Depth** | **Block Type**               | **Top-5 Error (ImageNet)** |
|---------------|-----------|------------------------------|-----------------------------|
| **ResNet-18** | 18        | Basic blocks (2 convs/block) | ~7.0%                       |
| **ResNet-34** | 34        | Basic blocks                 | ~5.7%                       |
| **ResNet-50** | 50        | **Bottleneck blocks** (1×1 → 3×3 → 1×1) | ~4.9%           |
| **ResNet-101**| 101       | Bottleneck blocks            | ~4.6%                       |
| **ResNet-152**| 152       | Bottleneck blocks            | ~4.4%                       |

---

## 📐 **Block Types**

### ✅ **1. Basic Block** (Used in ResNet-18, 34)
```text
Conv3x3 → BN → ReLU → Conv3x3 → BN → Add(skip) → ReLU
```

### ✅ **2. Bottleneck Block** (Used in ResNet-50+)
```text
Conv1x1 → BN → ReLU  
→ Conv3x3 → BN → ReLU  
→ Conv1x1 → BN → Add(skip) → ReLU
```

> The 1×1 convolutions reduce and then restore dimensions, making deep networks computationally efficient.

---

## 🧪 **ResNet-50 Architecture Outline**

| **Stage**     | **Output Size** | **Block Type**     | **# Blocks** |
|---------------|------------------|---------------------|--------------|
| Conv1         | 112×112×64       | 7×7 conv, stride 2   | –            |
| MaxPool       | 56×56×64         | 3×3, stride 2        | –            |
| Conv2_x       | 56×56×256        | Bottleneck           | 3            |
| Conv3_x       | 28×28×512        | Bottleneck           | 4            |
| Conv4_x       | 14×14×1024       | Bottleneck           | 6            |
| Conv5_x       | 7×7×2048         | Bottleneck           | 3            |
| Avg Pool      | 1×1×2048         | Global avg pooling   | –            |
| FC            | 1000             | Fully connected + softmax | –        |

![image](https://github.com/user-attachments/assets/a181c503-8be0-4afe-8a2f-01820af496f4)

---

## ✅ **Benefits of ResNet**

| Feature              | Benefit                                     |
|----------------------|---------------------------------------------|
| **Skip Connections** | Easier gradient flow; prevents degradation |
| **Very Deep Models** | Supports 50–150+ layers without overfitting|
| **Modular Design**   | Easy to adapt for detection, segmentation  |
| **Reusable Backbone**| Used in many models: Faster R-CNN, Mask R-CNN, etc. |

---

## 🔚 **Summary**

| **Aspect**         | **ResNet**                                |
|--------------------|--------------------------------------------|
| Year               | 2015                                       |
| Main Innovation    | Residual (skip) connections                |
| Top Accuracy       | 3.57% Top-5 error (ResNet-152, ImageNet)   |
| Variants           | ResNet-18, 34, 50, 101, 152                |
| Block Types        | Basic and Bottleneck                      |
| Legacy             | Backbone for modern CV models             |
