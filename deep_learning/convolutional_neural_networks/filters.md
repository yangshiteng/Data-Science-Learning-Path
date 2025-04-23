# **Filters (Kernels)** in CNNs

## **What is a Filter?**

A **filter**, or **kernel**, is a small matrix of numbers (weights) that is applied to the input image or feature map to extract specific features. These could be **edges**, **textures**, or more complex patterns.

> 💡 Think of a filter as a little pattern scanner that looks at small patches of the image and decides if a certain feature is present there.

---

## **Filter Structure**

- **Shape**: Usually 2D or 3D (depending on the input)
  - For a grayscale image: e.g., `3 × 3`
  - For a color image: e.g., `3 × 3 × 3` (since RGB has 3 channels)
- **Values**: Initially random, but they are **learned** during training using backpropagation.

---

## **How Filters Work**

Here’s what happens step-by-step:
1. The filter slides over the image with a certain **stride**.
2. At each position, it performs **element-wise multiplication** between the filter and the corresponding patch of the image.
3. It **sums up** the results into a single value.
4. This value becomes one pixel in the **output feature map**.

This operation is called **convolution operation**.

![image](https://github.com/user-attachments/assets/4e7ab9c2-ff41-4692-b786-dd76346056cd)

---

## **Why Use Multiple Filters?**

Each filter is designed to pick up on **a different pattern**. For example:
- One filter might detect **vertical edges**
- Another might detect **horizontal edges**
- Another might highlight **corners** or **curves**

When you stack multiple filters, you get **multiple feature maps**, giving the network a richer understanding of the image.

---

## **Example: Edge Detection Filter**

A simple 3×3 filter that detects horizontal edges might look like:

```
[[-1, -1, -1],
 [ 0,  0,  0],
 [ 1,  1,  1]]
```

When applied to an image, this filter highlights regions where the pixel intensity changes vertically — that is, where horizontal edges are present.

---

## **Filter Size and Depth**

- Common filter sizes: `3×3`, `5×5`, `7×7`
- For color images, the depth of the filter matches the input channels (e.g., `3×3×3` for RGB)

---

## **Learned, Not Handcrafted**

In early days, filters were manually designed (like edge detectors).  
But in CNNs:
- Filters start with random values
- The network **learns the optimal filter weights** during training to best capture the important features for the task (like classification)
