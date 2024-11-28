![image](https://github.com/user-attachments/assets/327a951e-9646-449f-ab22-9033af64695e)

![image](https://github.com/user-attachments/assets/9f686018-05d1-44fd-b22e-46a829ebb36b)

![image](https://github.com/user-attachments/assets/e7d3d60c-eae6-407c-900a-282c3fa33b8c)

![image](https://github.com/user-attachments/assets/94a9b90d-2ea7-43b0-bfd6-722e9d3b620d)

![image](https://github.com/user-attachments/assets/f14366fc-c0f2-4b4c-884c-de786dd60c9f)

![image](https://github.com/user-attachments/assets/3ab50bf0-dbd3-42aa-a28f-d186749b9144)

![image](https://github.com/user-attachments/assets/4dcbaa10-5d2f-4246-9dfe-41fc57911b62)

![image](https://github.com/user-attachments/assets/7cf17e90-dfe0-4316-8f0c-2873ed82ec9b)

```python
from sklearn.datasets import make_circles
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Create synthetic dataset (two concentric circles)
X, y = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=42)

# Visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.title("Non-Linearly Separable Data")
plt.show()

# Train Non-Linear SVM with RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma=0.5)
model.fit(X, y)

# Function to plot decision boundary
import numpy as np

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.title("Decision Boundary (RBF Kernel)")
    plt.show()

# Plot decision boundary
plot_decision_boundary(X, y, model)

# Predict a new observation
new_observation = [[0, 0]]
predicted_class = model.predict(new_observation)
print(f"Predicted class for {new_observation}: {predicted_class[0]}")

```

![image](https://github.com/user-attachments/assets/2ec16fe3-57df-4d6d-8a8e-08a650fa8597)

![image](https://github.com/user-attachments/assets/7759b557-8abb-4282-a7b8-1b7caca37735)

![image](https://github.com/user-attachments/assets/ace4618a-e1e8-48ee-b2ae-3e6b55f119a5)






