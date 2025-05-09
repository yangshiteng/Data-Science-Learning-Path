# **Introduction to Convolutional Neural Networks (CNNs)**

## **What is a CNN?**

A **Convolutional Neural Network (CNN)** is a type of deep learning model specifically designed to process data with a grid-like topology, such as images. Unlike traditional fully connected neural networks, CNNs use a mathematical operation called **convolution** in at least one of their layers, which allows them to automatically and adaptively learn spatial hierarchies of features from input data.

CNNs consist of layers such as:
- **Convolutional layers** – for feature extraction
- **Pooling layers** – for dimensionality reduction
- **Fully connected layers** – for final classification or output

![image](https://github.com/user-attachments/assets/0faa5a4b-269b-4bec-99bc-c2763a4ec552)

![image](https://github.com/user-attachments/assets/10d56da3-e529-4594-abc8-95a4c2b028b7)

This architecture makes CNNs especially powerful in visual data processing.

---

## **History and Motivation**

The concept of CNNs dates back to the 1980s and 1990s. Here's a brief timeline:

- **1980s**: Inspired by the structure of the visual cortex in animals, early models like the **Neocognitron** (proposed by Kunihiko Fukushima in 1980) laid the groundwork for modern CNNs.
- **1998**: Yann LeCun introduced **LeNet-5**, a CNN designed for handwritten digit recognition (e.g., MNIST dataset). It was one of the first successful applications of CNNs.
- **2012**: CNNs gained massive attention when **AlexNet**, developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) by a large margin. This success marked the start of the deep learning boom.

**Motivation** behind CNNs:
- Traditional neural networks struggle with high-dimensional image data due to scalability issues.
- CNNs exploit local spatial correlations and reduce the number of parameters through shared weights and local connectivity.
- This makes them both more **efficient** and **effective** for processing images.
