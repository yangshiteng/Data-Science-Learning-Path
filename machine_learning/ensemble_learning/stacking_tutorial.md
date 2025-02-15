![image](https://github.com/user-attachments/assets/8663d037-2c8f-48ba-9b23-3357644af15b)

# 📌 **Stacking in Ensemble Learning: A Step-by-Step Guide**
> 🚀 **A comprehensive guide to using Stacking for classification and regression in Python.**

---

## **1️⃣ Introduction**
**Stacking (Stacked Generalization)** is an **ensemble learning technique** that combines multiple machine learning models to **improve predictive performance**. Unlike **Bagging** and **Boosting**, which aggregate predictions in a simple way (e.g., averaging or weighted voting), **Stacking trains a meta-model (blender) to learn the best way to combine base models.**

### **✨ Why Use Stacking?**
✅ **Combines multiple models' strengths** – More robust and accurate predictions.  
✅ **Learns how to optimize the combination of models**.  
✅ **Reduces bias and variance by using diverse base learners**.  
✅ **Works for both classification and regression problems**.  

---

## **2️⃣ How Stacking Works**
Stacking involves **two levels of models**:

- **Level 0 (Base Models)**: A set of diverse machine learning models (e.g., Decision Tree, SVM, Random Forest).  
- **Level 1 (Meta-Model or Blender)**: A model that learns from the base models' predictions and makes the final decision.

### **📌 Stacking Process**
1. Train multiple **base models** on the training data.
2. Use these models to make **predictions** on a validation set.
3. Train a **meta-model (blender)** using these predictions as input features.
4. The meta-model makes the final prediction.

---

## **3️⃣ Installing Dependencies**
Make sure you have the required libraries installed:

```bash
pip install numpy pandas scikit-learn
```

Now, import the necessary libraries:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
```

---

## **4️⃣ Stacking for Classification**
### **📌 Step 1: Create a Synthetic Dataset**
```python
# Generate a synthetic binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### **📌 Step 2: Define Base Models**
```python
# Define base learners
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
]
```

---

### **📌 Step 3: Train Base Models and Generate Meta-Features**
```python
# Train base models and store predictions for training meta-model
meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
meta_features_test = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    model.fit(X_train, y_train)
    
    # Generate out-of-sample predictions
    meta_features_train[:, i] = model.predict(X_train)
    meta_features_test[:, i] = model.predict(X_test)
```

---

### **📌 Step 4: Train Meta-Model**
```python
# Define and train the meta-model (blender)
meta_model = LogisticRegression()
meta_model.fit(meta_features_train, y_train)
```

---

### **📌 Step 5: Make Final Predictions**
```python
# Meta-model makes final predictions
y_pred = meta_model.predict(meta_features_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Classification Accuracy: {accuracy:.4f}")
```

---

## **5️⃣ Stacking for Regression**
### **📌 Step 1: Generate Regression Data**
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### **📌 Step 2: Define Base Models**
```python
# Define base regressors
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
]
```

---

### **📌 Step 3: Train Base Models and Generate Meta-Features**
```python
# Train base models and store predictions for training meta-model
meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
meta_features_test = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    model.fit(X_train, y_train)
    
    # Generate out-of-sample predictions
    meta_features_train[:, i] = model.predict(X_train)
    meta_features_test[:, i] = model.predict(X_test)
```

---

### **📌 Step 4: Train Meta-Model**
```python
# Define and train the meta-model (blender)
meta_model = LinearRegression()
meta_model.fit(meta_features_train, y_train)
```

---

### **📌 Step 5: Make Final Predictions**
```python
# Meta-model makes final predictions
y_pred = meta_model.predict(meta_features_test)

# Compute Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Stacking Regression Mean Squared Error: {mse:.4f}")
```

---

## **6️⃣ Using Scikit-Learn’s StackingClassifier**
Scikit-Learn provides a built-in `StackingClassifier` for easy implementation.

```python
from sklearn.ensemble import StackingClassifier

# Define base models
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
]

# Define Stacking Classifier
stack_clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression(), cv=5)

# Train and evaluate
stack_clf.fit(X_train, y_train)
y_pred = stack_clf.predict(X_test)
print(f"StackingClassifier Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## **7️⃣ Summary**
✔ **Stacking is an ensemble technique that trains a meta-model to combine base models’ predictions.**  
✔ **Works well for both classification and regression tasks.**  
✔ **Improves accuracy by leveraging diverse base learners.**  
✔ **Scikit-Learn provides a built-in `StackingClassifier` for easy implementation.**  

