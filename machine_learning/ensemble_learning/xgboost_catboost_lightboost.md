# XGBoost

![image](https://github.com/user-attachments/assets/809933a7-4a6b-42c6-b453-221ff2aa0b24)

![image](https://github.com/user-attachments/assets/2c4b70f1-1f70-44df-93cd-975c39a28972)

```python
pip install xgboost
```

![image](https://github.com/user-attachments/assets/2eb4ccac-7de7-4f7f-a68a-867fb7a09cf2)

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression
```

![image](https://github.com/user-attachments/assets/efc8c779-9f8d-43b7-872e-3574ce19f449)

```python
# Create a synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

![image](https://github.com/user-attachments/assets/f7d134a2-1d61-42f3-bd30-0280990ad281)

```python
# Convert dataset into DMatrix (XGBoost's optimized data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
```

![image](https://github.com/user-attachments/assets/77f7338a-7a31-4b75-b5b2-07fcf138bd86)

```
params = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss',        # Log loss metric
    'max_depth': 4,                   # Depth of trees
    'learning_rate': 0.1,              # Step size shrinkage
    'n_estimators': 100,               # Number of boosting rounds
    'subsample': 0.8,                  # Random fraction of data used per tree
    'colsample_bytree': 0.8,           # Features used per tree
    'lambda': 1,                       # L2 regularization (Ridge)
    'alpha': 0.5,                      # L1 regularization (Lasso)
    'random_state': 42
}
```

![image](https://github.com/user-attachments/assets/b542c7f6-130d-4f5f-bc1b-a3d071502f18)

```python
# Train the XGBoost model
num_round = 100
watchlist = [(dtrain, 'train'), (dtest, 'test')]  # Track performance on test set
model = xgb.train(params, dtrain, num_boost_round=num_round, evals=watchlist, early_stopping_rounds=10)
```

![image](https://github.com/user-attachments/assets/f1d2cbc5-8c0b-4c3f-a54a-77ca1be3e2be)

![image](https://github.com/user-attachments/assets/26cdfaba-50f7-4df0-a96d-e60f851d843a)

```python
# Make predictions
y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary classes

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Classification Accuracy: {accuracy:.4f}")
```
![image](https://github.com/user-attachments/assets/6bdfeaee-e4b1-45b5-9a59-380b9c28581b)

![image](https://github.com/user-attachments/assets/d83e9b52-73e4-470c-b6d1-5563c125505a)

```python
import matplotlib.pyplot as plt

# Plot feature importance
xgb.plot_importance(model)
plt.show()
```
![image](https://github.com/user-attachments/assets/065bb18a-3914-4f32-bc0d-f114cde70909)




