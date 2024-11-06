![image](https://github.com/user-attachments/assets/0f75539c-177e-477a-99b6-7b8717086bec)

![image](https://github.com/user-attachments/assets/7a70fe5c-6c00-4187-b8e2-6f14b6b680dd)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_classification

# Generate synthetic data
X, _ = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, 
                           n_clusters_per_class=1, weights=[0.95], class_sep=2, random_state=42)

# Introduce some anomalies
X[-20:] = np.random.uniform(low=-4, high=4, size=(20, 2))

# Fit the model
clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
preds = clf.fit_predict(X)
scores = clf.decision_function(X)

# Plot the data and the regions where the anomalies are
plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
radius = (scores.max() - scores) / (scores.max() - scores.min())
plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/4ce7a00c-d658-4419-ad17-d35612acf84c)

![image](https://github.com/user-attachments/assets/45e67049-12d8-431f-95db-d0eef7e24874)
