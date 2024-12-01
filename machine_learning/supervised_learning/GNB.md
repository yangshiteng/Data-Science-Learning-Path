![image](https://github.com/user-attachments/assets/82e98526-87a8-4800-ace6-d34135026148)

![image](https://github.com/user-attachments/assets/35363e16-f1d2-4af5-8baf-ba7df8c94524)

![image](https://github.com/user-attachments/assets/8a057016-7070-45bf-8b12-4891551a9ecf)

![image](https://github.com/user-attachments/assets/061ddf9e-b6b3-4429-b775-8b9542ad104c)

![image](https://github.com/user-attachments/assets/0678b206-a42c-40fd-8e31-a0ac154099cc)

![image](https://github.com/user-attachments/assets/1bc85117-6586-4204-9abb-bd41fd3070bf)

![image](https://github.com/user-attachments/assets/b5444d75-b36e-4fdb-bc4b-9f8e8ae6fb83)

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Dataset
X_train = np.array([[5.0], [6.0], [7.0], [1.0], [2.0], [3.0]])  # Features
y_train = np.array([1, 1, 1, 0, 0, 0])  # Labels (1 = Class A, 0 = Class B)

# New data point to classify
X_test = np.array([[4.5]])

# Train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Log of probabilities (log-priors + log-likelihoods)
log_probs = gnb.predict_log_proba(X_test)
predicted_class = gnb.predict(X_test)

print(f"Log Probabilities: {log_probs}")
print(f"Predicted Class: {predicted_class[0]}")
```

![image](https://github.com/user-attachments/assets/ba2d5a06-ba68-4918-a030-1302694aaa27)

