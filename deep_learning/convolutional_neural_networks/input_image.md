# Input Image in CNNs

The **input image** is the raw data that a CNN receives at the very beginning of the network. It typically represents a visual scene, such as a handwritten digit, a face, a cat, or a street view. CNNs are designed to extract and learn patterns from this image automatically.

## **Structure of an Input Image**

An image is represented as a multi-dimensional array (tensor) of pixel values.

- **Grayscale image**:  
  - Shape: `Height × Width × 1`  
  - Each pixel has a single intensity value (0–255).
  
- **Color image (RGB)**:  
  - Shape: `Height × Width × 3`  
  - Each pixel has **three values** corresponding to Red, Green, and Blue channels.

![image](https://github.com/user-attachments/assets/eb1d93cb-abeb-4c54-8dbd-fcf63f6c9178)

> 📌 Example:  
A 28×28 grayscale image of a digit from the MNIST dataset will have shape:  
`28 × 28 × 1`  
A 224×224 color image from ImageNet will have shape:  
`224 × 224 × 3`

## **Pixel Values and Normalization**

- **Raw pixel values** typically range from 0 to 255.
- Before feeding them into the network, it's common to **normalize** these values to a range of 0–1 or -1 to 1, which helps speed up and stabilize training.

## **Input as a Tensor**

CNNs treat the image as a **tensor** with three dimensions:
```
Input Tensor Shape = (Height, Width, Channels)
```
In batch processing, an extra dimension is added for the number of samples:
```
Batch Input Shape = (Batch Size, Height, Width, Channels)
```

---

# **Why the Input Image Matters**

The structure and content of the input image determine:
- The number of parameters in the first convolutional layer
- The size and depth of resulting feature maps
- The computational cost of training
- The type of patterns that can be learned

---

## **Example Input: Cat Image**
Let’s say you feed a 128×128 RGB photo of a cat into a CNN:
- Shape: `128 × 128 × 3`
- Pixel values: Matrix of numbers from 0 to 255 in each channel
- CNN will start by detecting **edges**, **textures**, and gradually learn higher-level features like **fur**, **ears**, or **eyes**.
