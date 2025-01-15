![image](https://github.com/user-attachments/assets/8a5b114a-9806-42e6-8004-1af65cb4b7d5)

![image](https://github.com/user-attachments/assets/37b56c5d-a96b-4fdd-90e7-e31773fd0c21)

![image](https://github.com/user-attachments/assets/a8c9fda4-8286-4147-a9e4-ddff07ca2cd2)

![image](https://github.com/user-attachments/assets/43659c78-edfa-4046-9f7c-38a240f3d69f)

![image](https://github.com/user-attachments/assets/418e8379-6e0e-462f-bf78-380d45c4f5ae)

![image](https://github.com/user-attachments/assets/f426b9e4-c564-4b24-8da8-8a94384eb128)

```python
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define base models
log_clf = LogisticRegression()
tree_clf = DecisionTreeClassifier()
svc_clf = SVC(probability=False)

# Create a Voting Classifier (Hard Voting)
voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('dt', tree_clf),
        ('svc', svc_clf)
    ],
    voting='hard'
)

# Train and evaluate
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
print("Hard Voting Accuracy:", accuracy_score(y_test, y_pred))
```

![image](https://github.com/user-attachments/assets/e40265a1-4d3c-4ef8-b098-dfc76469809a)

```python
# Create a Voting Classifier (Soft Voting)
voting_clf_soft = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('dt', tree_clf),
        ('svc', SVC(probability=True))  # Enable probability output
    ],
    voting='soft'
)

# Train and evaluate
voting_clf_soft.fit(X_train, y_train)
y_pred_soft = voting_clf_soft.predict(X_test)
print("Soft Voting Accuracy:", accuracy_score(y_test, y_pred_soft))

```

![image](https://github.com/user-attachments/assets/f11f49ed-67f2-47b9-bee2-57c9bc8ff8fc)

![image](https://github.com/user-attachments/assets/4402e6d2-3937-408a-9077-6f767cfae963)

![image](https://github.com/user-attachments/assets/f171d04a-e13c-464e-b97a-f8f2f730ec39)

![image](https://github.com/user-attachments/assets/fdcfdc7b-c275-4de8-acd5-704eb1cc8da6)


