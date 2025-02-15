![image](https://github.com/user-attachments/assets/a43d943a-a693-4c22-859d-f82c94939def)

![image](https://github.com/user-attachments/assets/726dac68-629a-4a65-bba4-376ee18eea83)

![image](https://github.com/user-attachments/assets/33d2191c-da2e-48b2-bf73-40e59c050367)

![image](https://github.com/user-attachments/assets/73da9b54-ce28-4a4d-b305-e931d5d51ee2)

![image](https://github.com/user-attachments/assets/ef6dc135-e090-47d7-a695-a72561f999c3)

![image](https://github.com/user-attachments/assets/b5019c6e-a868-47ae-a805-c380eaa67534)

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
tree = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=42)
tree.fit(X, y)

# Visualize the Tree
plot_tree(tree, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)

```
![image](https://github.com/user-attachments/assets/6525f573-8543-408c-80ac-778d015ed5e6)
