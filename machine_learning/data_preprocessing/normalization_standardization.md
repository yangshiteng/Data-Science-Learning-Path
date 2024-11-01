# Introduction

![image](https://github.com/user-attachments/assets/8c423c39-7701-4c48-892a-e9934b05ab3c)

![image](https://github.com/user-attachments/assets/5aa6c5c3-0f7d-4f61-b7bf-c9289fff198f)

```python
import pandas as pd
age_data = pd.DataFrame({'age': [22, 25, 37, 59, 45, 18, 33]})
age_data['age_binned'] = pd.cut(age_data['age'], bins=[0, 20, 40, 60], labels=["0-20", "21-40", "41-60"])
print(age_data)
```
![image](https://github.com/user-attachments/assets/fce2352e-fd70-423c-8f93-20c5e4844dfd)

![image](https://github.com/user-attachments/assets/391d2745-ffc5-4933-95fd-d3f28f872e70)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

# Sample data
data = np.array([[100, 0.1],
                 [200, 0.15],
                 [300, 0.2],
                 [400, 0.25],
                 [500, 0.3]])

# Applying Min-Max Scaling
scaler_min_max = MinMaxScaler()
data_min_max_scaled = scaler_min_max.fit_transform(data)

# Applying Standard Scaling
scaler_std = StandardScaler()
data_std_scaled = scaler_std.fit_transform(data)

# Applying L2 Normalization
normalizer = Normalizer(norm='l2')
data_normalized = normalizer.fit_transform(data)

# Convert to DataFrame for better readability
df_transformed = pd.DataFrame({
    'Original Feature 1': data[:, 0],
    'Min-Max Scaled Feature 1': data_min_max_scaled[:, 0],
    'Standard Scaled Feature 1': data_std_scaled[:, 0],
    'L2 Normalized Feature 1': data_normalized[:, 0]
})

print(df_transformed)

```
![image](https://github.com/user-attachments/assets/87ad041c-9f07-4931-9790-8f8d9fa9d8cd)

# How Normalization and Standardization Help in Convergence

![image](https://github.com/user-attachments/assets/505d91cc-a231-4376-8250-4704d9d09f78)

![image](https://github.com/user-attachments/assets/8e6f9248-ba7a-4f9c-8879-310d17b58875)


