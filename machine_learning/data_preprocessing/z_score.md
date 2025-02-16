![image](https://github.com/user-attachments/assets/516fd80c-0362-4790-b1fd-b88333514cd4)

![image](https://github.com/user-attachments/assets/6d20e224-7f71-484a-8771-8b26675d90d0)

![image](https://github.com/user-attachments/assets/907dc2f9-6490-4a78-9c55-79e8fac90d9a)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Generate Sample Data (Including Outliers)
np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=100)  # Normal data
outliers = np.array([10, 100, 105])  # Extreme outliers
data_with_outliers = np.concatenate((data, outliers))

# Convert to DataFrame
df = pd.DataFrame(data_with_outliers, columns=["Value"])

# Compute Z-Scores
df["Z-Score"] = zscore(df["Value"])

# Define Outliers (Using |Z| > 3)
df["Outlier"] = df["Z-Score"].apply(lambda z: abs(z) > 3)

# Display Outliers
print(df[df["Outlier"]])

# Plot Data with Outliers Marked
plt.figure(figsize=(10,5))
plt.scatter(range(len(df)), df["Value"], color='blue', label='Normal Data')
plt.scatter(df[df["Outlier"]].index, df[df["Outlier"]]["Value"], color='red', label='Outliers', marker='o', s=100)
plt.axhline(df["Value"].mean(), color='green', linestyle='dashed', label='Mean')
plt.title("Outlier Detection Using Z-Score")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.show()
```

![image](https://github.com/user-attachments/assets/07ec1500-f991-4986-8ec9-e6401d03433b)

![image](https://github.com/user-attachments/assets/bef0cd7e-5707-4057-b637-2b6249e1bbb4)

![image](https://github.com/user-attachments/assets/fbe8e7b4-99d7-4d90-a49d-4758fcf5f1ec)
