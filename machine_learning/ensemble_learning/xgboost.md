![image](https://github.com/user-attachments/assets/809933a7-4a6b-42c6-b453-221ff2aa0b24)

![image](https://github.com/user-attachments/assets/58993510-afe2-499a-9fbf-aeacea4b7f17)

![image](https://github.com/user-attachments/assets/86eff47c-cb8e-4090-ae6f-08a6b9fa79e5)

![image](https://github.com/user-attachments/assets/e334216a-72f5-49cd-bd37-14728e1d3498)

![image](https://github.com/user-attachments/assets/0e17e6ed-1da8-4138-8bb8-dcaa0dbae031)

![image](https://github.com/user-attachments/assets/c76efd5e-9267-45d4-a4f9-393f1f28bb73)

![image](https://github.com/user-attachments/assets/726a301d-098f-494d-8ca5-391bb9c08609)

![image](https://github.com/user-attachments/assets/c3d9ab06-56a2-43f1-be3c-c3fa8884fccf)


# Import Dependencies

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression
```

# XGBoost for Classification

## Step 1: Create a Synthetic Dataset

```python
# Create a synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

## Step 2: Convert Data to DMatrix

```python
# Convert dataset into DMatrix (XGBoost's optimized data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
```
## Step 3: Define XGBoost Parameters

```python
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

## Step 4: Train the Model

```python
# Train the XGBoost model
num_round = 100
watchlist = [(dtrain, 'train'), (dtest, 'test')]  # Track performance on test set
model = xgb.train(params, dtrain, num_boost_round=num_round, evals=watchlist, early_stopping_rounds=10)
```

## Step 5: Make Predictions and Evaluate

```python
# Make predictions
y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary classes

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Classification Accuracy: {accuracy:.4f}")

```

## Step 6: Feature Importance Visualization

```python
import matplotlib.pyplot as plt

# Plot feature importance
xgb.plot_importance(model)
plt.show()
```

# XGBoost for Regression

## Step 1: Generate Regression Data

```python
# Create a synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert dataset into DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

```
## Step 2: Define XGBoost Parameters

```python
params_reg = {
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'rmse',            # Root Mean Squared Error
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1,
    'alpha': 0.5,
    'random_state': 42
}
```

## Step 3: Train the Model

```python
# Train the XGBoost model for regression
model_reg = xgb.train(params_reg, dtrain, num_boost_round=100, evals=[(dtrain, 'train'), (dtest, 'test')], early_stopping_rounds=10)
```

## Step 4: Make Predictions and Evaluate

```python
# Make predictions
y_pred = model_reg.predict(dtest)

# Compute Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"XGBoost Regression Mean Squared Error: {mse:.4f}")
```

# Hyperparameter Tuning in XGBoost

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Define XGBoost model
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Define parameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Perform Grid Search
grid_search = GridSearchCV(xgb_clf, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best parameters
print(f"Best Parameters: {grid_search.best_params_}")
```

# XGBoost with Sklearn API

## Classification Example

```python
from xgboost import XGBClassifier

# Initialize classifier
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

# Train model
xgb_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_clf.predict(X_test)
print(f"XGBoost (Sklearn API) Classification Accuracy: {accuracy_score(y_test, y_pred):.4f}")

```

## Regression Example

```python
from xgboost import XGBRegressor

# Initialize regressor
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

# Train model
xgb_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_reg.predict(X_test)
print(f"XGBoost (Sklearn API) Regression MSE: {mean_squared_error(y_test, y_pred):.4f}")

```












