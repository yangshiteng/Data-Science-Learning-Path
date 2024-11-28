![image](https://github.com/user-attachments/assets/7dbe6434-1c6e-47af-b1b5-ac12d4b27d05)

![image](https://github.com/user-attachments/assets/7023d91b-f022-4ccf-9723-d56778a8aaa5)

![image](https://github.com/user-attachments/assets/30ff6ec1-5c25-4105-9b87-33fcee8ad4dc)

![image](https://github.com/user-attachments/assets/21c43277-a6af-418f-af2d-4d17e60288d4)

![image](https://github.com/user-attachments/assets/420945ef-0a9a-4c92-9999-2afbfe4246f8)

![image](https://github.com/user-attachments/assets/f5eb6b73-2c41-4452-bc9c-03d5ab8bf58b)

![image](https://github.com/user-attachments/assets/497fe8af-2b8d-4052-9643-8f2b037645c4)

![image](https://github.com/user-attachments/assets/697da6ba-7cee-4826-a0f1-130990e978eb)

![image](https://github.com/user-attachments/assets/5fb227d2-44c3-4cba-9a37-544042dc2a88)

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load Iris dataset (binary classification: class 0 vs. class 1)
iris = datasets.load_iris()
X = iris.data[:100, :2]  # Use first two features for visualization
y = iris.target[:100]

# Convert labels to -1 and +1
y = np.where(y == 0, -1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Hard Margin SVM (C is very large)
hard_margin_svm = SVC(kernel='linear', C=1e6)
hard_margin_svm.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = hard_margin_svm.predict(X_test)
print("Hard Margin SVM Performance:")
print(classification_report(y_test, y_pred))

# Train Soft Margin SVM (C is small)
soft_margin_svm = SVC(kernel='linear', C=1.0)
soft_margin_svm.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = soft_margin_svm.predict(X_test)
print("Soft Margin SVM Performance:")
print(classification_report(y_test, y_pred))

# Plot Decision Boundary
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title(title)
    plt.show()

plot_decision_boundary(X_test, y_test, hard_margin_svm, "Hard Margin SVM")
plot_decision_boundary(X_test, y_test, soft_margin_svm, "Soft Margin SVM")

```
![image](https://github.com/user-attachments/assets/89cebda6-032b-4dc6-a9cb-2685c1244f17)

![image](https://github.com/user-attachments/assets/9fbc7387-255e-472e-a5be-3a954e011f25)

![image](https://github.com/user-attachments/assets/857ee11b-6006-473c-a0ed-6972e918a77e)

![image](https://github.com/user-attachments/assets/3db0b264-410a-4632-a056-746158f21d37)
