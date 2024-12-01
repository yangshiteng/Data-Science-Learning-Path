![image](https://github.com/user-attachments/assets/eec166b8-08ea-4559-88aa-0a0d37e7394c)

![image](https://github.com/user-attachments/assets/58cabe87-190c-43d0-b5f0-de5fecb9bb17)

![image](https://github.com/user-attachments/assets/15d5f4c5-0e7b-4dbd-ac2c-b5f1e4dc1b38)

![image](https://github.com/user-attachments/assets/db88d24c-8f7e-4063-b54a-5dceaa739c4f)

![image](https://github.com/user-attachments/assets/70bdff46-99e9-4584-9529-c122478dbfce)

![image](https://github.com/user-attachments/assets/56e1b7c5-16f0-4a90-8197-0d6446555ce8)

![image](https://github.com/user-attachments/assets/eb40f4b3-c5df-44bd-9ab0-6e434fe11c70)

![image](https://github.com/user-attachments/assets/5081aad5-95bf-412f-9b9d-5f8b6c511069)

![image](https://github.com/user-attachments/assets/2f5fa8f9-c6c1-40f5-8c90-e71939707bcf)

![image](https://github.com/user-attachments/assets/a5785276-4911-4d07-b2d5-cf645a8d30cb)

```python

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Data
documents = ["win win game", "computer computer technology"]
labels = [1, 0]  # 1 = Sports, 0 = Technology

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# New document
X_new = vectorizer.transform(["win computer"])

# Train Multinomial Naive Bayes
mnb = MultinomialNB(alpha=1)
mnb.fit(X, labels)

# Predict
predicted_class = mnb.predict(X_new)
predicted_probabilities = mnb.predict_proba(X_new)

print("Predicted Class:", predicted_class[0])
print("Predicted Probabilities:", predicted_probabilities)

```

![image](https://github.com/user-attachments/assets/38d5dbba-15cf-4fbc-b28e-f54d8c44ccf5)
