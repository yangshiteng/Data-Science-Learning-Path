![image](https://github.com/user-attachments/assets/d8535eb6-eeb4-40a6-b8b2-69f60fdd7174)

![image](https://github.com/user-attachments/assets/98abe81f-976e-46d5-b79b-939748ca5e48)

![image](https://github.com/user-attachments/assets/8c1843e7-10ea-49f5-960c-664ae2ca6123)

![image](https://github.com/user-attachments/assets/0fae5acf-9b7f-4335-a3f1-87b6673ef6f4)

![image](https://github.com/user-attachments/assets/fe28e719-a7ce-4cb3-b161-af627683105d)

![image](https://github.com/user-attachments/assets/82490cd2-b9f5-4c78-8d7b-dc77529053c8)

![image](https://github.com/user-attachments/assets/c8122a29-a6f9-49e6-986f-517547fe7500)

![image](https://github.com/user-attachments/assets/00ec55f6-0625-4517-abc0-36f8d85682a0)

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bagging Classifier with Decision Trees
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=10,  # Number of trees
    bootstrap=True,  # Enable bootstrap sampling
    random_state=42
)

# Train and evaluate
bagging_clf.fit(X_train, y_train)
y_pred = bagging_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```
![image](https://github.com/user-attachments/assets/f130c7ae-1d87-41c5-85f3-7ab6fcb3895d)
