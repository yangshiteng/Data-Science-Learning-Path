![image](https://github.com/user-attachments/assets/d767fb33-d0c7-4e10-b52c-a5ddf96f2bd3)

![image](https://github.com/user-attachments/assets/c7666f75-2b5e-430a-8060-c179d20a6798)

# Python Example for Classification Tree

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Target

# Train a Decision Tree Classifier
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X, y)

# Get feature importance
feature_importances = model.feature_importances_

# Create a DataFrame for better readability
feature_importance_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(8, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Decision Tree')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()
```

![image](https://github.com/user-attachments/assets/c39dc524-ccac-4bd2-a343-7e1a0b433d69)

# Python Example for Regression Tree

```python
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Train Decision Tree Regressor
regressor = DecisionTreeRegressor(criterion='squared_error', max_depth=5, random_state=42)
regressor.fit(X, y)

# Get feature importances
feature_importances = regressor.feature_importances_

# Create a DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Decision Tree Regressor')
plt.gca().invert_yaxis()
plt.show()

```
![image](https://github.com/user-attachments/assets/c0273f93-88ff-4594-a1ca-55722e2076cb)




