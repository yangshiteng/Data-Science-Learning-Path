![image](https://github.com/user-attachments/assets/932eaeaf-2fec-4469-b23b-f0a4d51566dc)

![image](https://github.com/user-attachments/assets/5f39c2db-8f22-4f75-8f45-ef4f3ff5905c)

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize Logistic Regression with class weight 'balanced'
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Evaluation
print(classification_report(y_test, predictions))
```
![image](https://github.com/user-attachments/assets/5c59e3e8-7a26-4aa9-8b61-919fe48a33f0)
