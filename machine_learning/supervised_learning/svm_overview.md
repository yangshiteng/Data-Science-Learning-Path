![image](https://github.com/user-attachments/assets/cfaee24a-dd20-47e6-a934-d4202e476cfd)

![image](https://github.com/user-attachments/assets/39395731-682f-4d41-bd46-38dbcc400523)

![image](https://github.com/user-attachments/assets/6cdd335d-7412-40aa-b062-da9c9ba849cb)

![image](https://github.com/user-attachments/assets/d12a7ed5-993f-402d-a535-d86473f413d0)

![image](https://github.com/user-attachments/assets/67bd712a-66be-4a95-971b-0e319dc8edc8)

![image](https://github.com/user-attachments/assets/6f0b19b6-8e98-43e5-ae45-fa0f72b35490)

![image](https://github.com/user-attachments/assets/477fa588-0444-498b-be85-8b60ccbac7bc)

![image](https://github.com/user-attachments/assets/5023fefc-eda1-4e2e-a3ed-5b4dcba8d854)

![image](https://github.com/user-attachments/assets/b736183a-3857-443e-908c-6a3ba4a9675b)

![image](https://github.com/user-attachments/assets/34792054-f94d-4b6d-a648-7f79c9b03942)

# Python Implementation

## Example 1: Linear SVM

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binary classification (e.g., class 0 vs. rest)
y = (y == 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM model
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title("Non-linear SVM with RBF Kernel")
plt.show()

```
![image](https://github.com/user-attachments/assets/0de6c118-7edd-444a-ae84-b0972ed4240c)


## Example 2: Non-linear SVM with RBF Kernel

```python
from sklearn.datasets import make_circles
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create dataset
X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM model
model = SVC(kernel='rbf', C=1, gamma=0.5)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title("Non-linear SVM with RBF Kernel")
plt.show()

```

![image](https://github.com/user-attachments/assets/b69a91d9-c87c-4189-acad-613c4c6cfc01)

![image](https://github.com/user-attachments/assets/0672fc82-3f33-4176-b28a-10857baff4af)

![image](https://github.com/user-attachments/assets/04d5a85b-8141-4ae3-b961-076deb154a6d)





