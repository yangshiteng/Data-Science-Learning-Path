# 📈 **Time Series Forecasting with RNNs**

**Time series forecasting** is the task of predicting future values based on previously observed data points over time. This type of data appears in many fields — such as stock prices, weather conditions, sales figures, electricity demand, and sensor readings.

Traditional forecasting methods like ARIMA, exponential smoothing, or Holt-Winters work well for simple, linear relationships, but they struggle with complex patterns, long-term dependencies, and nonlinear dynamics.

That’s where **Recurrent Neural Networks (RNNs)** come in.

---

# 🧠 **Why RNNs for Time Series?**

RNNs are specifically designed to handle sequential data, where the **order** of inputs matters. Unlike standard feedforward neural networks, RNNs have loops that allow information to persist across time steps, giving them a form of memory.

This makes RNNs especially suited for:
✅ Capturing temporal dependencies
✅ Learning from long histories of data
✅ Handling variable-length input sequences

---

# 🔧 **How Does Time Series Forecasting Work with RNNs?**

1️⃣ **Input**: A series of past observations (e.g., temperature readings over the past 30 days).

2️⃣ **RNN Processing**: The RNN takes each time step’s input and passes information forward, updating its hidden state to summarize both the current input and past context.

3️⃣ **Output**: The model predicts the next time step(s) — either a single future point (one-step forecast) or multiple points ahead (multi-step forecast).

---

# 🛠 **Common RNN Architectures for Time Series**

* **Simple RNNs**: Basic loops but struggle with long-term dependencies (due to vanishing gradients).
* **LSTM (Long Short-Term Memory)**: Specialized RNNs that use gates to better capture long-range patterns.
* **GRU (Gated Recurrent Unit)**: A streamlined version of LSTM, often faster and similarly effective.

---

# 📦 **Applications**

✅ **Finance** → Stock price prediction, risk modeling
✅ **Energy** → Forecasting electricity demand, solar/wind output
✅ **Healthcare** → Predicting patient vitals, disease progression
✅ **Retail** → Sales forecasting, inventory planning
✅ **Weather** → Temperature, rainfall, and storm forecasting

---

# ⚠️ **Challenges**

* Needs large amounts of data for training
* Sensitive to noise and outliers
* Harder to interpret compared to statistical models
* Requires careful handling of seasonality and trends

---

# 🌟 **Example: Predicting Daily Temperatures**

Imagine you have a dataset with daily temperatures over 365 days, and you want to predict **tomorrow’s temperature** based on the past 7 days.

---

## 🏗 **Step 1: Prepare the Data**

We break the data into input–output pairs:

| Input (past 7 days)           | Output (next day) |
| ----------------------------- | ----------------- |
| \[21, 22, 23, 24, 22, 21, 22] | 23                |
| \[22, 23, 24, 22, 21, 22, 23] | 24                |
| \[23, 24, 22, 21, 22, 23, 24] | 25                |
| ...                           | ...               |

We **slide a window** over the data to create many such sequences.

---

## 🏃 **Step 2: Build the RNN Model**

We define a **simple RNN** (or LSTM/GRU) model:

* Input layer → receives sequences of length 7
* RNN layer → processes time dependencies
* Dense layer → outputs a single predicted temperature

---

## 🔧 **Step 3: Define the Loss Function**

Since we’re predicting a number, we use **Mean Squared Error (MSE)** as the loss:

$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

This measures how close the predicted temperature $\hat{y}_i$ is to the actual temperature $y_i$.

---

## 🏋️ **Step 4: Train the Model**

The training loop works like this:

1. **Forward pass**: Pass input sequences through the RNN to get predictions.
2. **Calculate loss**: Compare predictions to actual next-day temperatures using MSE.
3. **Backward pass**: Use backpropagation through time (BPTT) to compute gradients.
4. **Update weights**: Adjust model weights using an optimizer like Adam or SGD.

This process is repeated over **many epochs** (passes over the full dataset) until the model learns to minimize the prediction error.

---

## 💡 **Simple Pseudocode**

```python
for epoch in range(num_epochs):
    for X_batch, y_batch in data_batches:
        predictions = model(X_batch)
        loss = compute_mse(predictions, y_batch)
        loss.backward()  # backpropagation through time
        optimizer.step()  # update weights
```

---

## 📊 **Step 5: Evaluate**

After training:

* Test the model on unseen data (e.g., last 30 days).
* Compare predicted vs. actual temperatures.
* Optionally plot results to visualize performance.

---

## ✅ **Summary of Training Process**

| Step         | Purpose                                |
| ------------ | -------------------------------------- |
| Prepare data | Create sequences and targets           |
| Build model  | Define RNN layers                      |
| Define loss  | Choose how to measure prediction error |
| Train        | Optimize weights to minimize loss      |
| Evaluate     | Test generalization on new data        |
