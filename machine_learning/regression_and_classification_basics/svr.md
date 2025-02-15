![image](https://github.com/user-attachments/assets/f71ba477-ed26-4275-b8a4-efc2d12662b9)

![image](https://github.com/user-attachments/assets/c3f84d28-535a-4534-97e5-1e65ac56d1ba)

![image](https://github.com/user-attachments/assets/18ba0e07-0a37-40f5-a9f0-158e0958ef5f)

![image](https://github.com/user-attachments/assets/d2c03272-5d04-4b86-ae2a-892e136c2d15)

![image](https://github.com/user-attachments/assets/9f3333e4-03bd-433e-b129-d9df4eb07bd8)

![image](https://github.com/user-attachments/assets/9528040b-4d3e-417a-818f-8ee3a8e8d764)

```python
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVR with RBF kernel
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_train, y_train)

# Predict on test data
y_pred = svr_rbf.predict(X_test)

# Visualize SVR results
plt.scatter(X, y, color='darkorange', label='Data')
plt.plot(X, svr_rbf.predict(X), color='navy', lw=2, label='SVR Model (RBF Kernel)')
plt.legend()
plt.title("Support Vector Regression (RBF Kernel)")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

![image](https://github.com/user-attachments/assets/dc923f05-cb5c-414f-8be2-53a8761ef1bd)

![image](https://github.com/user-attachments/assets/8b0e4b6a-7920-4519-b506-a0daa112d43b)

```python
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 1],
    'gamma': [0.01, 0.1, 1]
}

# Perform grid search
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)
```
![image](https://github.com/user-attachments/assets/270a4cdb-53de-4001-8b4a-23c2059dd939)

![image](https://github.com/user-attachments/assets/907ed53f-996c-473a-8d28-af108af5c86e)

![image](https://github.com/user-attachments/assets/b9346f16-2474-4b5e-8818-b16c752e30b2)

![image](https://github.com/user-attachments/assets/1ee37dbc-8405-41f5-9eca-ef5ccf0bc5c8)

![image](https://github.com/user-attachments/assets/b4dfd4ae-020b-44fe-9240-0a84836cbade)




