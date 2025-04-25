## 🧠 **Inception v4 – Modular Deep CNN**

### 📌 **Overview**

| Feature              | Description                                      |
|----------------------|--------------------------------------------------|
| **Name**             | Inception v4                                     |
| **Published**        | 2016 (by Szegedy et al.)                         |
| **Paper**            | *Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning* |
| **Input Size**       | 299 × 299 × 3                                    |
| **Total Parameters** | ~43 million                                      |
| **Top-5 Error (ImageNet)** | ~3.08%                                     |

---

## 🏗️ **Inception v4 Architecture (Layer-by-Layer Breakdown)**

Your image shows the **main modules** of Inception v4 and their **data shapes**. Let's break it down:

| **Stage**         | **Block**           | **Description**                                       | **Output Shape**         |
|-------------------|---------------------|--------------------------------------------------------|---------------------------|
| **Input**         | –                   | 299×299×3 RGB image                                    | 299×299×3                 |
| **Stem**          | Conv + pooling      | Initial feature extraction block                       | 35×35×384                 |
| **Inception-A × 4**| Module A            | Multi-branch 1×1, 3×3, 5×5 filters + concat            | 35×35×384                 |
| **Reduction-A**   | Grid reduction      | Reduces spatial size, increases depth                 | 17×17×1024                |
| **Inception-B × 7**| Module B            | Deeper branches with asymmetric convs (1×7 + 7×1)     | 17×17×1024                |
| **Auxiliary Classifier** | Mid-network classifier (used during training only) | Classification assist and regularization | 17×17×1024                |
| **Reduction-B**   | Grid reduction      | Aggressive downsampling                               | 8×8×1536                  |
| **Inception-C × 3**| Module C            | Final Inception module with high-dimensional features | 8×8×1536                  |
| **Global Avg Pool**| Pooling             | 8×8 → 1×1 average pooling                              | 1×1×1536                  |
| **Dropout**       | Regularization      | Dropout (usually ~40%)                                | 1×1×1536                  |
| **FC Layer**      | Dense + softmax     | Fully connected → output softmax                      | 1000 (ImageNet classes)   |

---

## 🔍 **Understanding the Modules**

### ✅ **Stem**
- A stack of convolutions and pooling
- Converts input to a 35×35×384 representation

### ✅ **Inception-A**
- Captures multi-scale features
- Each block contains:
  - 1×1 conv
  - 3×3 conv
  - Two 3×3 convs in series
  - Avg pool + 1×1 conv
- Outputs are concatenated

### ✅ **Reduction-A**
- Shrinks spatial resolution to 17×17
- Mix of stride-2 conv and pooling

### ✅ **Inception-B**
- Uses **asymmetric convolutions** (1×7 then 7×1)
- Allows deep and wide receptive fields with fewer parameters

### ✅ **Reduction-B**
- Another downsampling layer to 8×8

### ✅ **Inception-C**
- Deepest block, captures highly abstract features
- Uses multiple 1×1 and 3×3 filters in various configurations

### ✅ **Auxiliary Classifier**
- Optional classifier (only used during training)
- Helps gradient flow, reduces overfitting

### ✅ **Final Classification**
- Global average pooling → Dropout → Fully connected → Softmax

---

![image](https://github.com/user-attachments/assets/2d45e41c-7adb-4e92-b92a-da95042d4583)

![image](https://github.com/user-attachments/assets/789053f6-46aa-42cf-9f45-ccaaa802e850)

![image](https://github.com/user-attachments/assets/85603e50-5462-4d23-81e9-8acfe0f8ccc5)

![image](https://github.com/user-attachments/assets/88e2e90f-f6f5-42d8-ae3e-d47d71af673b)

![image](https://github.com/user-attachments/assets/8c42e5e1-ff93-4201-a098-cd4e6dfa3d86)

![image](https://github.com/user-attachments/assets/2b9481dc-6778-4bfe-82ae-f67332e2d554)

![image](https://github.com/user-attachments/assets/0352fcf9-97d9-4f45-949c-dc802623d695)

![image](https://github.com/user-attachments/assets/2d4782b2-353c-4947-b2c9-8170d5b4fd83)

## 📈 **Performance & Characteristics**

| Metric              | Value                |
|---------------------|----------------------|
| Input Size          | 299×299×3            |
| Top-5 Accuracy      | ~96.9% (Top-5 error ~3.08%) |
| Parameters          | ~42–43 million       |
| Depth               | Deeper than v3       |
| Global Pooling      | Yes                  |
| Residuals           | ❌ Not in v4 (used in Inception-ResNet) |

---

## ✅ **Summary Table**

| **Component**       | **Repetitions** | **Shape**              |
|---------------------|------------------|--------------------------|
| Stem                | –                | 35×35×384                |
| Inception-A         | 4×               | 35×35×384                |
| Reduction-A         | 1×               | 17×17×1024               |
| Inception-B         | 7×               | 17×17×1024               |
| Reduction-B         | 1×               | 8×8×1536                 |
| Inception-C         | 3×               | 8×8×1536                 |
| Global Pooling + FC | –                | 1×1×1536 → 1000          |
