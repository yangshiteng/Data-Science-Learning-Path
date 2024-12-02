![image](https://github.com/user-attachments/assets/56b9991e-62cc-4a8f-914e-22e4110501ce)

![image](https://github.com/user-attachments/assets/8d25a2a1-b1bc-4bc9-b1ec-e4e532e7b196)

![image](https://github.com/user-attachments/assets/640a3889-66cf-4068-bcc8-400acf9cb6ec)

![image](https://github.com/user-attachments/assets/fdc1ebbf-520d-48ad-80a3-931a74eea9be)

![image](https://github.com/user-attachments/assets/0ec56ae7-46d6-411d-90c8-c5a0cfbd701e)

![image](https://github.com/user-attachments/assets/bc71089c-2455-4a1e-9710-d7fcdedfec5c)

![image](https://github.com/user-attachments/assets/5c78730c-1adf-4e13-9692-5acc4c095bde)

![image](https://github.com/user-attachments/assets/5c853933-3473-4cb3-a32f-7bf7f8815599)

![image](https://github.com/user-attachments/assets/cdd7d23b-6e96-4869-be54-1cba0a47871f)

# Python Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

# Data
documents = ["win game", "computer technology"]
labels = [1, 0]  # 1 = Sports, 0 = Technology

# Transform documents into binary feature vectors
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(documents)

# New document
X_new = vectorizer.transform(["win computer"])

# Train Bernoulli Naive Bayes
bnb = BernoulliNB(alpha=1)
bnb.fit(X, labels)

# Predict
predicted_class = bnb.predict(X_new)
predicted_probabilities = bnb.predict_proba(X_new)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Predicted Class:", predicted_class[0])
print("Predicted Probabilities:", predicted_probabilities)
```
![image](https://github.com/user-attachments/assets/3b4185e5-5e3b-4913-97c4-4952f67fab5d)
