![image](https://github.com/user-attachments/assets/06d07474-693f-46c1-8949-c38ead4c94f7)

![image](https://github.com/user-attachments/assets/4495028f-4129-4961-b4c3-9b280df861d0)

![image](https://github.com/user-attachments/assets/a88fc031-2da4-49a6-85b6-f21da16bf467)

![image](https://github.com/user-attachments/assets/d75731d9-f198-4783-aee6-05aa4bca1816)

![image](https://github.com/user-attachments/assets/c6dd2f59-796b-41bd-9a39-9376e8a3e061)

![image](https://github.com/user-attachments/assets/84adc5fc-d9a4-49cb-80fc-6d414c2d9a30)

![image](https://github.com/user-attachments/assets/9cd15d84-069f-4954-947a-d7935aa02336)

![image](https://github.com/user-attachments/assets/acefdfb6-38a5-43fe-b491-4a7bbc56fd8f)

![image](https://github.com/user-attachments/assets/87cf165c-9b40-45b1-99d9-efeb7021a2bc)

![image](https://github.com/user-attachments/assets/e8b0abe6-cfc3-402c-967d-da60c95ed9b9)

![image](https://github.com/user-attachments/assets/a992411f-dd79-47ec-b630-34cc738be376)

![image](https://github.com/user-attachments/assets/80c6e9ad-dfc9-4c33-bcd5-4e7be3779228)

![image](https://github.com/user-attachments/assets/f3ba7a92-ed8d-42ef-b271-764844e03548)

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(16, activation='relu', input_shape=(20,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

![image](https://github.com/user-attachments/assets/6e44ff42-6171-4287-b278-056598fa4169)

![image](https://github.com/user-attachments/assets/f861934c-7d96-4884-aa8e-600cb3a6b567)

![image](https://github.com/user-attachments/assets/e97e335c-e086-404b-b2b4-f631b8896980)

![image](https://github.com/user-attachments/assets/63b3ad1b-7ccf-4f73-a04c-4e10a71d9fd2)

![image](https://github.com/user-attachments/assets/1199f0e4-83a6-482c-b300-2d718d99bb82)
