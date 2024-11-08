![image](https://github.com/user-attachments/assets/ae3e2650-3e1a-4aa4-96cb-ae428197466e)

![image](https://github.com/user-attachments/assets/600dac9d-743b-4c98-8e24-1c64caa0786e)

![image](https://github.com/user-attachments/assets/74b93092-324a-4a5d-ac96-3ba041f51b65)

# Python Example - Outlier Detection with IQR

```python
import numpy as np

data = np.random.randn(100) * 20 + 20  # Normal data
data = np.append(data, [200, -100])  # Add some clear outliers

Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1

outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
```
# Python Example - Anomaly Detection with Isolation Forest

```
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

data, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.5, random_state=42)
data = np.r_[data, np.random.uniform(low=-10, high=10, size=(20, 2))]  # Add potential anomalies

clf = IsolationForest(contamination=0.1)
preds = clf.fit_predict(data)
```
# Python Example - Novelty Detection with One Class SVM

```python
import numpy as np
from sklearn.svm import OneClassSVM

X_train = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X_train + 2, X_train - 2]

clf = OneClassSVM(nu=0.1)
clf.fit(X_train)

X_test = np.random.uniform(low=-4, high=4, size=(20, 2))
test_preds = clf.predict(X_test)
```
