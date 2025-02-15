![image](https://github.com/user-attachments/assets/f3c5c233-b1a8-4913-9df3-b2fb8ebd3e7f)

![image](https://github.com/user-attachments/assets/3975bc10-7149-494f-86ae-2ff5dd0df0ae)

![image](https://github.com/user-attachments/assets/fa1f8c28-a400-4609-a3e2-0d3b4f5d6e59)

![image](https://github.com/user-attachments/assets/e19382ae-36a8-4760-ba6a-92217bfbea34)

![image](https://github.com/user-attachments/assets/2f14ad8c-4327-425a-9586-ae3065836590)

![image](https://github.com/user-attachments/assets/1d703bce-01f0-4ff9-b4a1-941075a241aa)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Generate synthetic data: 100 houses mainly clustered with 20 isolated ones
np.random.seed(42)
clustered_houses = np.random.randn(100, 2)  # Main cluster
isolated_houses = np.random.uniform(low=-10, high=10, size=(20, 2))  # Isolated houses
houses = np.vstack([clustered_houses, isolated_houses])

# Apply Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=3, contamination=0.1)
outlier_flags = lof.fit_predict(houses)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(houses[:, 0], houses[:, 1], color='k', s=50, label='Houses', alpha=0.6)
outliers = houses[outlier_flags == -1]
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', s=50, label='Outliers')
plt.title('Local Outlier Factor Detecting Isolated Houses')
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/33367be1-7ec6-4967-a464-4c66c23ed2cc)
