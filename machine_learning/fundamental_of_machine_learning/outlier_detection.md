![image](https://github.com/user-attachments/assets/aa3123bd-4e5d-4050-be9d-d164dd2e2881)

![image](https://github.com/user-attachments/assets/62a6479c-aca2-4f95-925b-3894fc3771f5)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs

# Generate synthetic data with clusters
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Add some random noise as outliers
X = np.r_[X, np.random.uniform(low=-10, high=10, size=(20, 2))]

# Fit the LOF model
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
X_scores = clf.negative_outlier_factor_

# Plot the level of outlier-ness
plt.title("Local Outlier Factor (LOF)")
plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
# Circle out potential outliers
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
outlier_scores = plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
                             facecolors='none', label='Outlier scores')
plt.axis('tight')
plt.xlim((-10, 10))
plt.ylim((-10, 10))
plt.xlabel("Prediction errors: %d" % (y_pred != 1).sum())
plt.legend([outlier_scores], ['Outlier scores'], loc='upper left')
plt.show()
```

![image](https://github.com/user-attachments/assets/86684539-2415-486a-b838-f6f86c7d6586)

![image](https://github.com/user-attachments/assets/fcb2b0f6-4dee-47b4-83f6-41a5790a106e)
