![image](https://github.com/user-attachments/assets/6a4c5044-263c-4eb2-ba6f-36834c36a5f3)

![image](https://github.com/user-attachments/assets/928eae11-d06d-4998-ab05-391d64e5e0b6)

![image](https://github.com/user-attachments/assets/12aa1533-491e-4eff-bd51-a264ce3ccc16)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# Generate synthetic data
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

# Visualize original data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
colors = ['red' if v == 0 else 'blue' for v in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, label='Original', edgecolor='k', alpha=0.5)
plt.title('Original Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Apply SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Visualize the data after SMOTE
plt.subplot(1, 2, 2)
colors = ['red' if v == 0 else 'blue' for v in y_res]
plt.scatter(X_res[:, 0], X_res[:, 1], c=colors, label='SMOTE', edgecolor='k', alpha=0.5)
plt.title('Data Distribution after SMOTE')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Output the class distribution before and after
print("Class distribution before SMOTE:", np.bincount(y))
print("Class distribution after SMOTE:", np.bincount(y_res))
```
![image](https://github.com/user-attachments/assets/6114ac5c-9c88-4e5c-9209-cf8ce632ffa4)

![image](https://github.com/user-attachments/assets/7be859cd-cac2-4e1e-b881-bc5d54cb10c9)

