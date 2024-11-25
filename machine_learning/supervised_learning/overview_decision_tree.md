![image](https://github.com/user-attachments/assets/c5572a10-9aff-4078-a333-ea461fca0120)

![image](https://github.com/user-attachments/assets/adb7fdf7-7cea-4690-b0fc-2d14a77f3a79)

![image](https://github.com/user-attachments/assets/7fd1c906-082b-48b3-9cde-8019f077fcad)

![image](https://github.com/user-attachments/assets/63858020-30e4-4228-afeb-ff97ca90835b)

![image](https://github.com/user-attachments/assets/844e9bb4-43c3-4475-b41c-b02f22fdc1a9)

![image](https://github.com/user-attachments/assets/a5dbb356-1720-47ce-9ac7-1d1082c087fe)

![image](https://github.com/user-attachments/assets/16c294da-211b-42bf-840c-d58a4169ef33)

# Python Implementation for Classification Tree

```python
# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the Decision Tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()

```

![image](https://github.com/user-attachments/assets/a4d5f591-0650-4beb-b3ab-fa1ab880aa3e)

# Python Implementation for Regression Tree

```python
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor
regressor = DecisionTreeRegressor(criterion='squared_error', max_depth=5, random_state=42)
regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(regressor, feature_names=data.feature_names, filled=True)
plt.show()

```

![image](https://github.com/user-attachments/assets/9eb5a471-8faa-4acf-ab8f-380620a39236)

![image](https://github.com/user-attachments/assets/63ab7b3b-f7b8-4411-a650-c048151e8e59)

