![image](https://github.com/user-attachments/assets/6eeb36a1-1bae-4c0b-b4b6-0766834c7f6b)

![image](https://github.com/user-attachments/assets/f63c7835-3f12-48e2-b46b-e5f5038411fc)

![image](https://github.com/user-attachments/assets/f3152fb6-4f8a-4589-9b38-e8c164a85dba)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate Sample Data (Including Outliers)
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=100)  # Normally distributed data
outliers = np.array([10, 100, 105])  # Extreme outliers
data_with_outliers = np.concatenate((data, outliers))

# Convert to DataFrame
df = pd.DataFrame(data_with_outliers, columns=["Value"])

# Compute Q1, Q3, and IQR
Q1 = df["Value"].quantile(0.25)  # First Quartile (25th percentile)
Q3 = df["Value"].quantile(0.75)  # Third Quartile (75th percentile)
IQR = Q3 - Q1  # Compute Interquartile Range

# Define Outlier Bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify Outliers
df["Outlier"] = (df["Value"] < lower_bound) | (df["Value"] > upper_bound)

# Display Outliers
print("Outliers Detected:\n", df[df["Outlier"]])

# Plot Data with Outliers Marked
plt.figure(figsize=(10,5))
plt.scatter(range(len(df)), df["Value"], color='blue', label='Normal Data')
plt.scatter(df[df["Outlier"]].index, df[df["Outlier"]]["Value"], color='red', label='Outliers', marker='o', s=100)
plt.axhline(Q1, color='green', linestyle='dashed', label='Q1 (25th Percentile)')
plt.axhline(Q3, color='purple', linestyle='dashed', label='Q3 (75th Percentile)')
plt.axhline(lower_bound, color='red', linestyle='dashed', label='Lower Bound (Outlier Threshold)')
plt.axhline(upper_bound, color='red', linestyle='dashed', label='Upper Bound (Outlier Threshold)')
plt.title("Outlier Detection Using IQR Method")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.show()
```

![image](https://github.com/user-attachments/assets/b56599d4-6165-4f41-a00d-7567d21938c7)

![image](https://github.com/user-attachments/assets/158246c0-0f74-4617-94e2-d4b6080662c6)

![image](https://github.com/user-attachments/assets/c3504198-fb3e-4c3d-9513-68fbd7679dfd)






