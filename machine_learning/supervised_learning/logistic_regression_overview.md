![image](https://github.com/user-attachments/assets/be101e22-38ea-4415-adf5-090bf59fa9d3)

![image](https://github.com/user-attachments/assets/4ec544b0-b976-4cb1-b458-309fab14ffa0)

![image](https://github.com/user-attachments/assets/8abd408b-a175-45d1-a78f-6dc9bd9fc07e)

![image](https://github.com/user-attachments/assets/8d2f35ca-4693-447b-ad7f-847159517e21)

![image](https://github.com/user-attachments/assets/32ebc20b-ac78-403b-b2c2-7c9750ac552c)

![image](https://github.com/user-attachments/assets/42746dea-2743-471b-9970-d1ca0077ca04)

![image](https://github.com/user-attachments/assets/0bbd6c9b-008e-4b9c-8aa4-4c778056d070)

![image](https://github.com/user-attachments/assets/d38a815e-87c5-49ac-bdd2-f4f9d285df46)

![image](https://github.com/user-attachments/assets/9935b563-7582-4395-8101-f195518d46f2)

![image](https://github.com/user-attachments/assets/d938786b-d340-4808-9e86-f96a673b7d64)

![image](https://github.com/user-attachments/assets/f58e0a2a-d205-4963-9a13-5053d77d2486)

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate a synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```
![image](https://github.com/user-attachments/assets/9cd6dc36-83d5-49ea-821f-f0a65cb45a1b)
