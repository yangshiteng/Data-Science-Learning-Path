![image](https://github.com/user-attachments/assets/0dbb7e00-784d-4854-9166-367f2f90010d)

![image](https://github.com/user-attachments/assets/f9d9257e-28e0-4fee-8bb5-25bb680704fb)

![image](https://github.com/user-attachments/assets/50e1649b-8477-4c0f-8000-d520b40b5736)

```python
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Generate a synthetic imbalanced dataset
X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.1, 0.9],  # 10% positive, 90% negative
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=1000, random_state=10)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Applying SMOTE to generate synthetic samples
print("Before SMOTE, counts of label '1': {}".format(sum(y_train==1)))
print("Before SMOTE, counts of label '0': {} \n".format(sum(y_train==0)))

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE, counts of label '1': {}".format(sum(y_train_smote==1)))
print("After SMOTE, counts of label '0': {}".format(sum(y_train_smote==0)))

# Training a classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_smote, y_train_smote)
y_pred = clf.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
```
![image](https://github.com/user-attachments/assets/b28a1e6b-a2a3-4957-91e4-7b66c85a8c1d)

![image](https://github.com/user-attachments/assets/182a8b9a-c841-4e55-b842-4986a955b260)






