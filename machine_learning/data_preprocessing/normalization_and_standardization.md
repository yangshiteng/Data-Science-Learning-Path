# Introduction

![image](https://github.com/user-attachments/assets/075b4e54-ec10-4208-b2ee-229b0dde0d23)

![image](https://github.com/user-attachments/assets/63bdf3b1-26c4-4874-9682-635c32928ab5)

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# Sample data
data = np.array([[50, 2], [60, 4], [70, 6]])

# Applying Min-Max Normalization
scaler_min_max = MinMaxScaler()
data_normalized = scaler_min_max.fit_transform(data)

# Applying Standardization
scaler_standard = StandardScaler()
data_standardized = scaler_standard.fit_transform(data)

print("Normalized Data:\n", data_normalized)
print("Standardized Data:\n", data_standardized)

```
![image](https://github.com/user-attachments/assets/83407f96-ea3e-4f43-99d7-521b8a8c96ed)

# How Normalization and Standardization Help in Convergence

![image](https://github.com/user-attachments/assets/505d91cc-a231-4376-8250-4704d9d09f78)

![image](https://github.com/user-attachments/assets/8e6f9248-ba7a-4f9c-8879-310d17b58875)
