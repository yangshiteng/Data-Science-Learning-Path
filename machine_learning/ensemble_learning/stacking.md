![image](https://github.com/user-attachments/assets/72929a50-6506-4fb7-8f64-62645a33b973)

![image](https://github.com/user-attachments/assets/2ef085ac-1db7-43d0-b96a-73c3ef749fee)

![image](https://github.com/user-attachments/assets/26e95c11-94c6-4603-b5fc-154202c0f929)

![image](https://github.com/user-attachments/assets/557537e2-3f8f-4d7f-a606-818bf1bc9aa9)

![image](https://github.com/user-attachments/assets/10c2168c-39e4-48fa-b3ae-ed7e038f3fa3)

![image](https://github.com/user-attachments/assets/bd3de7c5-f79c-4b80-9fd5-df23dfc5e036)

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]

# Define the meta-model
meta_model = LogisticRegression()

# Create the StackingClassifier
stack_clf = StackingClassifier(
    estimators=base_models,  # Base models
    final_estimator=meta_model,  # Meta-model
    cv=5  # Cross-validation for predictions
)

# Train and evaluate
stack_clf.fit(X_train, y_train)
y_pred = stack_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

```
![image](https://github.com/user-attachments/assets/83d598c6-15e0-4278-84ea-6f003398941f)

![image](https://github.com/user-attachments/assets/6c3d65fb-c382-4abe-b223-3e847ae62b4a)

![image](https://github.com/user-attachments/assets/66a3f102-3841-4fc4-b4ae-097f5280b2c6)

