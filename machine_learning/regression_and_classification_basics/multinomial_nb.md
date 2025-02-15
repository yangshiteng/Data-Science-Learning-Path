![image](https://github.com/user-attachments/assets/982d01a9-befb-4e1f-9416-15435ca13512)

![image](https://github.com/user-attachments/assets/bee1a0d9-5aec-4f2b-8caf-cc518159448d)

![image](https://github.com/user-attachments/assets/4baba5fa-ce91-40ba-a041-687578c9b8ad)

![image](https://github.com/user-attachments/assets/ce26db04-d207-4b94-849f-3b58989fa348)

![image](https://github.com/user-attachments/assets/7a3b9474-eb3b-4487-8667-44fcf34e6241)

![image](https://github.com/user-attachments/assets/11472bd4-e7cc-4130-8bdf-183b12050b2a)

![image](https://github.com/user-attachments/assets/66728a7d-32e5-43ee-ad64-7a2b7228ba96)

![image](https://github.com/user-attachments/assets/bbf0941e-0a83-41ec-9cd7-edf8afbaf7d2)

# Python Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Data
documents = ["win game", "computer technology"]
labels = [1, 0]  # 1 = Sports, 0 = Technology

# Transform documents into feature vectors
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

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Predicted Class:", predicted_class[0])
print("Predicted Probabilities:", predicted_probabilities)
```
![image](https://github.com/user-attachments/assets/dc1d991d-3a47-47ea-9e5a-2a701e9e42a7)
