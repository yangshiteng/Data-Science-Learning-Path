![image](https://github.com/user-attachments/assets/952ef757-a33e-4bf5-a4ba-e4f62f32df38)

![image](https://github.com/user-attachments/assets/714b9308-6cda-4c29-a501-76a2d29d1293)

![image](https://github.com/user-attachments/assets/0a4f00c3-7f28-4ba3-90c2-14481f52daf1)

![image](https://github.com/user-attachments/assets/71b151c5-bbcc-48b1-a353-37980d496b39)

![image](https://github.com/user-attachments/assets/4d19fdde-bd69-48b3-9026-c472b5c7312a)

![image](https://github.com/user-attachments/assets/3d31ba76-dead-4ac8-88f1-0caa77a71a5e)

![image](https://github.com/user-attachments/assets/f2b17209-6bd8-46a3-be47-c1cd086bb4ad)

![image](https://github.com/user-attachments/assets/173579e2-bbca-4e45-82dc-f68c3536a612)

![image](https://github.com/user-attachments/assets/bf7d54b1-e28c-4c02-b0f3-e8b16741d6b0)

![image](https://github.com/user-attachments/assets/d9e63d7d-a483-4add-a11f-162125be44c5)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Train the model
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))

```

![image](https://github.com/user-attachments/assets/514b944f-449b-4760-8a9d-8eb956cca221)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# Create synthetic dataset
X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# Train the model
rf_reg.fit(X_train, y_train)

# Make predictions
y_pred = rf_reg.predict(X_test)

# Evaluate performance
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
```

![image](https://github.com/user-attachments/assets/e6935081-15e8-4c70-83f6-786108e1777b)

```python
import matplotlib.pyplot as plt
import numpy as np

# Get feature importance
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure(figsize=(10,6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.title("Feature Importance in Random Forest")
plt.show()
```

![image](https://github.com/user-attachments/assets/8f50edaf-eded-430d-bb8a-68fae17473f1)

![image](https://github.com/user-attachments/assets/b2f3a3dd-3de2-4474-be37-ef48aadc439a)



