![image](https://github.com/user-attachments/assets/72f14bf8-f99a-4c3b-9dfd-a0c0b5ce6cbd)

![image](https://github.com/user-attachments/assets/88fd622c-2fd8-41bc-9be7-a3e15155ac15)

![image](https://github.com/user-attachments/assets/7aca2674-f304-4c14-ab6b-cb1d1e3a9b97)

![image](https://github.com/user-attachments/assets/cc27dfff-fbf9-4b8f-9d4e-2a13446a9af7)

![image](https://github.com/user-attachments/assets/de439a19-b36a-4acb-91ce-8cf326be5aae)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predictions
y_pred = gnb.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
![image](https://github.com/user-attachments/assets/dc4542d6-e535-49fe-9f37-cb0a003b74e3)

![image](https://github.com/user-attachments/assets/e2998c26-54f2-4dda-9644-74c716a5c5c4)

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Sample data
messages = [
    "Free entry in a weekly competition! Text WIN to 12345",
    "Hey, how are you doing today?",
    "Congratulations, you've won a free ticket to Bahamas!",
    "Call me when you get this.",
    "Win cash prizes by entering now!"
]
labels = [1, 0, 1, 0, 1]  # 1 = Spam, 0 = Not Spam

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train Multinomial Naive Bayes
nb = MultinomialNB(alpha=1)  # Laplace smoothing
nb.fit(X_train, y_train)

# Predictions
y_pred = nb.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
![image](https://github.com/user-attachments/assets/fc0e27ce-5eae-4734-a4c3-acebfa942d79)

![image](https://github.com/user-attachments/assets/bc3fee8e-00bc-40d1-afef-8e52dfab43cc)

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample data
messages = [
    "Free entry in a weekly competition! Text WIN to 12345",
    "Hey, how are you doing today?",
    "Congratulations, you've won a free ticket to Bahamas!",
    "Call me when you get this.",
    "Win cash prizes by entering now!"
]

labels = [1, 0, 1, 0, 1]  # 1 = Spam, 0 = Not Spam

# Convert text to binary feature vectors
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(messages)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train Bernoulli Naive Bayes
bnb = BernoulliNB(alpha=1)
bnb.fit(X_train, y_train)

# Predictions
y_pred = bnb.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
![image](https://github.com/user-attachments/assets/b0a2c203-b0d5-4e17-9d82-92849f13e97c)

![image](https://github.com/user-attachments/assets/4b33eedb-4c1c-4649-8746-e2789b35b824)

![image](https://github.com/user-attachments/assets/6f9178cb-9c73-4d93-b72a-c176bce348f9)







