![image](https://github.com/user-attachments/assets/514d5c98-a8eb-4c92-92d3-985c07b070d8)

![image](https://github.com/user-attachments/assets/0793546b-3f2c-4637-b08b-d31ffd81b82a)

![image](https://github.com/user-attachments/assets/ad6e4168-ed85-4568-821c-b456e30083ec)

![image](https://github.com/user-attachments/assets/4d33d045-e241-4523-ba6d-11a79b11ced8)

![image](https://github.com/user-attachments/assets/85816122-004e-4221-a48d-33b5a434ea80)

![image](https://github.com/user-attachments/assets/e43731b7-dc98-44d0-ba05-055fb6a81c7a)

![image](https://github.com/user-attachments/assets/e172ff51-ab94-4c18-a3eb-fc76ebc20c24)

![image](https://github.com/user-attachments/assets/9d8772b3-78d4-4f4a-9a56-8173b938e52b)

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

# Data
documents = ["win game", "computer technology"]
labels = [1, 0]  # 1 = Sports, 0 = Technology

# Convert text to binary feature vectors
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(documents)

# Display binary representation
print("Feature Names:", vectorizer.get_feature_names_out())
print("Binary Feature Matrix:\n", X.toarray())

# New document
X_new = vectorizer.transform(["win computer"])

# Train Bernoulli Naive Bayes
bnb = BernoulliNB(alpha=1)
bnb.fit(X, labels)

# Predict
predicted_class = bnb.predict(X_new)
predicted_probabilities = bnb.predict_proba(X_new)

print("Predicted Class:", predicted_class[0])
print("Predicted Probabilities:", predicted_probabilities)
```
![image](https://github.com/user-attachments/assets/cfdc6b1f-e820-4e87-ac17-b8beebf08dfb)

![image](https://github.com/user-attachments/assets/70e39077-40ef-4cc0-a5aa-b7b139417984)
