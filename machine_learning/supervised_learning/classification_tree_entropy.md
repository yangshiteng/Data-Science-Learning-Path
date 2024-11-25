![image](https://github.com/user-attachments/assets/f636326c-6138-45fa-a37c-887ae5e255bc)

![image](https://github.com/user-attachments/assets/f3d56c13-4890-46d0-8d0d-c7c12724783b)

![image](https://github.com/user-attachments/assets/616fd294-d047-48ff-bb93-87dd1c918d82)

![image](https://github.com/user-attachments/assets/800fab3c-ae30-4bbc-ad34-9146539044b5)

![image](https://github.com/user-attachments/assets/0977c349-641a-4a27-b2e9-70c5cf46050d)

![image](https://github.com/user-attachments/assets/9cde056b-26b8-431c-9d89-d2a78a67b807)

# Python Implementation

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Dataset
data = {
    'Feature1': ['High', 'High', 'Medium', 'Low', 'Low'],
    'Feature2': ['Low', 'High', 'Low', 'Medium', 'Low'],
    'Class': ['Yes', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Convert categorical variables to numerical values
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_encoded.drop('Class_Yes', axis=1)
y = df_encoded['Class_Yes']

# Train Decision Tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=42)
tree.fit(X, y)

# Visualize the Tree
plot_tree(tree, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
```
![image](https://github.com/user-attachments/assets/6934d55e-556c-42cb-b9b4-5d9e3bbc6758)
