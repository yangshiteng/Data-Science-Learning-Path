![image](https://github.com/user-attachments/assets/71addf4e-401d-40cd-b285-4b2172ccfd00)

![image](https://github.com/user-attachments/assets/ba4d0b58-fa06-45b2-8dad-0367d42e4d69)

![image](https://github.com/user-attachments/assets/32d4f083-ef24-4564-9cbb-0c62367ba3a5)

![image](https://github.com/user-attachments/assets/bb916df1-c99f-457e-810b-476169e5e652)

![image](https://github.com/user-attachments/assets/04e19b76-6955-42d4-89b8-93ced24b1374)

![image](https://github.com/user-attachments/assets/98ce3aee-a876-4554-a092-8c963bbca319)

![image](https://github.com/user-attachments/assets/c0bb73b6-8139-4b43-9343-94debbbd7f61)

![image](https://github.com/user-attachments/assets/5432e082-42d3-4d23-a2ca-51c0655e017a)

![image](https://github.com/user-attachments/assets/da2d6bf4-7cb3-4912-a42b-92e51195c629)

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize AdaBoost Classifier with Decision Trees as weak learners
adaboost_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Decision stumps
    n_estimators=50,  # Number of weak learners
    learning_rate=1.0,
    random_state=42
)

# Train and evaluate
adaboost_clf.fit(X_train, y_train)
y_pred = adaboost_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

![image](https://github.com/user-attachments/assets/2c06c89f-8d9a-4bb8-b11b-9f50b55a6845)
