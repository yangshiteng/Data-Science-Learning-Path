# Introduction

![image](https://github.com/user-attachments/assets/4d0f3363-7011-4207-af3f-4a9ad39071e2)

![image](https://github.com/user-attachments/assets/7c7af7e7-da76-4f05-8d43-1655b4f63d43)

![image](https://github.com/user-attachments/assets/c1b78196-a9da-4f24-8e46-16ac216925c2)

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# Sample data
data = np.array([[10, 2.7, 3.6],
                 [15, 3.6, 14.0],
                 [16, 2.3, 15.2]]).astype(np.float64)

# Normalize data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
print("Normalized Data:")
print(normalized_data)

# Standardize data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
print("\nStandardized Data:")
print(standardized_data)
```
# How Normalization and Standardization Help in Convergence

![image](https://github.com/user-attachments/assets/505d91cc-a231-4376-8250-4704d9d09f78)

![image](https://github.com/user-attachments/assets/8e6f9248-ba7a-4f9c-8879-310d17b58875)


