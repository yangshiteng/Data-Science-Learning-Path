![image](https://github.com/user-attachments/assets/f5b450bf-d31c-4126-8b1f-1cbfc6f6c11b)

![image](https://github.com/user-attachments/assets/e5742c7d-d006-45cb-84ba-5d6a3c469fe9)

# Example Setup

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.95, 0.05], flip_y=0,
                           random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

# Implementing AdaBoost

```python
# Initialize AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=100, random_state=42)

# Train the model
ada.fit(X_train, y_train)

# Predict and evaluate
ada_preds = ada.predict(X_test)
print("AdaBoost Classifier Report:\n", classification_report(y_test, ada_preds))
```
![image](https://github.com/user-attachments/assets/82f50e6d-6244-49a0-83ce-6ee16c912bab)

# Implementing Random Forest

```python
# Initialize RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
rf.fit(X_train, y_train)

# Predict and evaluate
rf_preds = rf.predict(X_test)
print("Random Forest Classifier Report:\n", classification_report(y_test, rf_preds))
```
![image](https://github.com/user-attachments/assets/44da0c31-1293-470b-a01a-2f6b5d2ea033)

![image](https://github.com/user-attachments/assets/81d16b95-b889-484c-b2a3-67895a2c53fc)

