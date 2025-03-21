![image](https://github.com/user-attachments/assets/cad76df0-87fe-4725-b1e4-c546b5351434)

![image](https://github.com/user-attachments/assets/4b80acb4-3b48-4e67-8dbb-1de3d1161ed2)

![image](https://github.com/user-attachments/assets/771c109d-0737-4413-b8a1-178d1ebee1a6)

![image](https://github.com/user-attachments/assets/518c6dec-0ded-4a5e-9692-8d6cccc38796)

![image](https://github.com/user-attachments/assets/cf3b2944-92c4-4d9d-b69c-18be2afd0958)

![image](https://github.com/user-attachments/assets/1d0480a1-b78b-4a24-b9bc-1211551d1bbf)

```python
from sklearn.datasets import load_iris
data = load_iris()
X, y = data.data, data.target
```

![image](https://github.com/user-attachments/assets/d8c9c2c7-9244-498b-a9b8-e6b291f3a8d5)

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

![image](https://github.com/user-attachments/assets/c45fb3a2-4044-4cab-beb5-2ebd7b812ce8)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

![image](https://github.com/user-attachments/assets/8af30035-8210-4799-bbb4-840d47dac0e4)

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)  # Set K=3
knn.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/1fd1bc5b-4371-43e8-89e6-63fb85e10594)

```python
y_pred = knn.predict(X_test)
```
![image](https://github.com/user-attachments/assets/c9e97864-ee92-47cb-ace2-d76c78293814)

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
![image](https://github.com/user-attachments/assets/6a7e15bb-a655-4432-adae-047892e58117)

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

data = load_iris()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

k_values = range(1, 21)
accuracy_scores = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    accuracy_scores.append(scores.mean())

plt.plot(k_values, accuracy_scores)
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Choosing Optimal K")
plt.show()
```
![image](https://github.com/user-attachments/assets/bcb63394-0f5f-4a85-9b42-43a49b528800)

![image](https://github.com/user-attachments/assets/6b1419ba-b958-4299-b485-5bc1337d7507)

![image](https://github.com/user-attachments/assets/9feef79c-b9d5-471a-a4ea-9a6f0e8901d8)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = load_iris()
X, y = data.data, data.target

# Preprocess Data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
![image](https://github.com/user-attachments/assets/627831e2-4c14-4375-b4f5-9443bd5127c1)





