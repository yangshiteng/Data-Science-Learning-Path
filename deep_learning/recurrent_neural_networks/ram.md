# 👁️‍🗨️ **Recurrent Attention Models (RAM)**

---

## 📘 **What Are Recurrent Attention Models?**

A **Recurrent Attention Model (RAM)** is a neural architecture that **processes input sequentially and selectively**, attending to only **parts of the input at each time step**, rather than the full input. It's inspired by how humans look at complex scenes — focusing on one part at a time.

> Instead of processing the entire input all at once, a RAM **learns where to look**, and **when to look**.

RAMs are particularly useful for **high-dimensional inputs**, like images, videos, and long text, where processing the entire input at full resolution is expensive.

---

## 🧠 **Key Idea**

* The model uses an RNN to **control attention over time**.
* At each time step:

  1. It **selects a region** (or part) of the input to attend to.
  2. Processes that region using a neural network.
  3. Updates an internal **state** (e.g., memory or belief).
* Eventually, it makes a **prediction** based on the sequence of glimpses.

---

## 🧱 **Architecture Components**

1. **Glimpse Sensor**

   * Extracts a **localized patch** (or region) of the input (e.g., a cropped image region)
   * The patch can vary in size or resolution (like a retina)

2. **Glimpse Network**

   * Processes the patch (e.g., using CNN for images or embedding layers for text)

3. **Core RNN**

   * Maintains internal state (e.g., via LSTM or GRU)
   * Guides attention location over time

4. **Location Network**

   * Predicts the **next attention location** (possibly stochastically)

5. **Action Network**

   * Outputs the final **classification or prediction**

6. **Baseline Network** (optional)

   * Used to reduce variance during training (for REINFORCE-based training)

---

## 🔄 **How It Works (at Time Step $t$)**

![image](https://github.com/user-attachments/assets/f0b7c80b-7885-4ba8-a82b-59cfc9358b23)

---

## 🎯 **Why Use RAM?**

| Problem                               | How RAM Helps                             |
| ------------------------------------- | ----------------------------------------- |
| High-dimensional input (e.g., images) | Focuses computation on salient parts      |
| Long sequences                        | Reduces processing to most relevant parts |
| Interpretability                      | Attention trace shows what the model saw  |
| Resource efficiency                   | Processes fewer pixels/tokens per step    |

---

## 🧰 **Applications**

| Domain                       | RAM Use Case                                          |
| ---------------------------- | ----------------------------------------------------- |
| 👁️ Image Recognition        | Classify image by looking at parts sequentially       |
| 📝 Document Reading          | Focus on key words/sentences in long text             |
| 🎥 Video Processing          | Attend to key frames or regions over time             |
| 🧠 Visual Question Answering | Look at specific parts of image to answer questions   |
| 🎮 Reinforcement Learning    | Learn to observe important state features dynamically |

---

## 🧪 **Training**

* Often trained using **reinforcement learning** (e.g., REINFORCE algorithm), since the **location selection is non-differentiable**
* Loss is typically a combination of:

  * Task loss (e.g., classification cross-entropy)
  * Policy gradient loss for learning attention policy
  * Baseline loss to stabilize training

---

## 🔧 Example (Image Classification)

* Instead of processing the full image:

  * The RAM first looks at the **top left** (patch)
  * Then the **bottom right**
  * Then the **center**
  * Aggregates this information and predicts the class

> Efficient and interpretable — you know exactly **what the model focused on**.

---

## 🧾 Summary

| Component       | Description                              |
| --------------- | ---------------------------------------- |
| Attention style | Hard attention (explicit location-based) |
| Core module     | RNN (e.g., LSTM) to manage state         |
| Selection       | Sequential glimpse of input              |
| Benefit         | Efficiency + interpretability            |
| Challenge       | Requires reinforcement learning to train |
