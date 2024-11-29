![image](https://github.com/user-attachments/assets/2785b94b-908c-4164-9bdf-be82b23c8ea6)

![image](https://github.com/user-attachments/assets/ccdf97ae-02e8-4f35-bd37-93ea78bf6fb0)

![image](https://github.com/user-attachments/assets/8b3857d2-ff38-408f-9fb3-874e5f6b7ae3)

![image](https://github.com/user-attachments/assets/7964408c-7082-4f7a-a98a-f937d61edbaf)

![image](https://github.com/user-attachments/assets/99a182e4-0921-40c3-911b-c9d08301dcf4)

![image](https://github.com/user-attachments/assets/ce954c25-2e0b-4409-8f50-cb68653b8c6d)

![image](https://github.com/user-attachments/assets/fc7ac283-ae9d-4267-8969-00e3129ede1b)

```python
from sklearn.datasets import make_moons
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Generate a non-linear dataset
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Train SVM models with different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
models = []
for kernel in kernels:
    model = SVC(kernel=kernel, gamma='scale', C=1)
    model.fit(X, y)
    models.append(model)

# Plot decision boundaries for each kernel
def plot_decision_boundary(model, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    ax.set_title(title)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, kernel in enumerate(kernels):
    plot_decision_boundary(models[i], X, y, axes[i // 2, i % 2], f"Kernel: {kernel}")
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/95c7b17b-7f2f-45c4-9b5f-66e4a8644bcd)



