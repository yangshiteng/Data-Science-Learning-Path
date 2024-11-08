![image](https://github.com/user-attachments/assets/44d98f99-c9b5-4c58-b869-b55be905501c)

![image](https://github.com/user-attachments/assets/04164bbd-e576-4207-8695-63b1bb2cd26e)

![image](https://github.com/user-attachments/assets/31809333-06d3-4abc-b8f7-566069f8747c)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate some synthetic data
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]  # 200 points total
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))  # 20 outliers

# Create the Isolation Forest model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)

# Predict anomalies
y_pred_train = clf.predict(X_train)
y_pred_outliers = clf.predict(X_outliers)

# Plot
plt.scatter(X_train[:, 0], X_train[:, 1], color='black', s=30, label='Training points')
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], color='red', s=30, label='Outliers')
plt.legend()
plt.title('Isolation Forest Anomaly Detection')
plt.show()
```
![image](https://github.com/user-attachments/assets/fb15b281-0d0d-43d0-9307-a1e685fdfa5c)

![image](https://github.com/user-attachments/assets/0ad8eb72-aad6-449d-9752-0f02c735964c)
