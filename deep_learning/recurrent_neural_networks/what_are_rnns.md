**Recurrent Neural Networks (RNNs)** are a class of artificial neural networks designed for processing **sequential data**. Unlike traditional feedforward neural networks, RNNs have **loops** that allow information to persist, making them especially effective for tasks where context or order matters (e.g., language, time series, audio).

![image](https://github.com/user-attachments/assets/241ca432-8f2f-4682-a9e9-e6b2904bffdc)

# Key Features of RNNs:

* **Memory of Past Inputs**: RNNs maintain a hidden state that captures information from previous time steps, allowing the model to "remember" prior inputs.
* **Shared Weights Across Time**: The same weights are applied at every time step, which helps in generalizing across sequences of varying lengths.
* **Sequential Processing**: RNNs handle inputs one step at a time, making them suitable for variable-length sequences.

# Applications:

* Language modeling and generation
* Machine translation
* Speech recognition
* Time series forecasting
* Music composition

# Limitations:

* **Vanishing and exploding gradients** during training, which makes learning long-range dependencies difficult.
* Difficulty in parallel computation due to sequential processing.

To address these issues, variants like **Long Short-Term Memory (LSTM)** and **Gated Recurrent Units (GRU)** were developed, which are better at capturing long-term dependencies.
