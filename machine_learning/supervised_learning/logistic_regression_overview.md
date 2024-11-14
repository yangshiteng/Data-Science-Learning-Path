![image](https://github.com/user-attachments/assets/626a2e4a-048e-4355-9000-316baadd34d1)

![image](https://github.com/user-attachments/assets/ed841af0-fe1b-4d7c-8210-cd813e17f185)

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate a synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```
![image](https://github.com/user-attachments/assets/6f6b8277-8aab-436d-a3ca-57ff5a09fffc)
