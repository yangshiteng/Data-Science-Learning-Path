![image](https://github.com/user-attachments/assets/1e3c4ebc-2331-42d4-a4af-64be90552452)

![image](https://github.com/user-attachments/assets/6e6f69e2-86d4-4847-a87e-eafc87e4fb71)

![image](https://github.com/user-attachments/assets/594ce76e-47e2-49e3-af83-5e5f991321e6)

![image](https://github.com/user-attachments/assets/cb0a2756-5bee-4b0a-a991-f0c755b699c1)

![image](https://github.com/user-attachments/assets/1e9ec993-c45c-4231-9b2a-4ecef4fd985f)

![image](https://github.com/user-attachments/assets/c3f49f63-da0a-4ea0-99f3-a89bc233295d)

![image](https://github.com/user-attachments/assets/736e30a5-cb14-486a-8144-c46f813deea3)

![image](https://github.com/user-attachments/assets/ba8e893a-1627-47c1-858a-ab3b48b5d2bf)

![image](https://github.com/user-attachments/assets/19ff3ebd-f449-440e-affc-388a5fa83816)

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




