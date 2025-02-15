### 📌 **CatBoost Tutorial: A Step-by-Step Guide**
> 🚀 **A comprehensive guide to using CatBoost for classification and regression in Python.**

---

## **1️⃣ Introduction**
**CatBoost (Categorical Boosting)** is a high-performance **gradient boosting algorithm** developed by **Yandex**. It is particularly useful for datasets with **categorical features** and supports both **classification** and **regression** tasks.

### **✨ Why Use CatBoost?**
✅ **Handles categorical features natively** (no need for encoding).  
✅ **Faster training** (ordered boosting).  
✅ **Less hyperparameter tuning required**.  
✅ **Supports GPU acceleration**.  
✅ **Prevents overfitting with built-in regularization**.

---

## **2️⃣ Installation & Importing Dependencies**
### **🔹 Install CatBoost**
```bash
pip install catboost
```

### **🔹 Import Libraries**
```python
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression
import matplotlib.pyplot as plt
```

---

## **3️⃣ CatBoost for Classification**
### **📌 Step 1: Create a Synthetic Dataset**
```python
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### **📌 Step 2: Define and Train a CatBoost Classifier**
```python
cat_clf = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=50,
    random_seed=42
)

cat_clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
```

---

### **📌 Step 3: Make Predictions and Evaluate**
```python
y_pred = cat_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"CatBoost Classification Accuracy: {accuracy:.4f}")
```

---

### **📌 Step 4: Feature Importance Visualization**
```python
feature_importances = cat_clf.get_feature_importance()
plt.figure(figsize=(10,5))
plt.bar(range(len(feature_importances)), feature_importances, color='blue')
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.title("Feature Importance in CatBoost")
plt.show()
```

---

## **4️⃣ CatBoost for Regression**
### **📌 Step 1: Generate Regression Data**
```python
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### **📌 Step 2: Define and Train a CatBoost Regressor**
```python
cat_reg = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=50,
    random_seed=42
)

cat_reg.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
```

---

### **📌 Step 3: Make Predictions and Evaluate**
```python
y_pred = cat_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"CatBoost Regression Mean Squared Error: {mse:.4f}")
```

---

## **5️⃣ Handling Categorical Features**
CatBoost can **directly handle categorical features** without encoding.

### **📌 Step 1: Prepare Data with Categorical Features**
```python
data = pd.DataFrame({
    'feature_1': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
    'feature_2': [10, 20, 10, 30, 20, 30, 10, 20],
    'target': [1, 0, 1, 0, 1, 0, 1, 0]
})

categorical_features = [0]  # First column is categorical
```

---

### **📌 Step 2: Convert Data into CatBoost Pool**
```python
train_data = Pool(data.drop(columns=['target']), label=data['target'], cat_features=categorical_features)
```

---

### **📌 Step 3: Train Model with Categorical Features**
```python
cat_clf = CatBoostClassifier(iterations=100, verbose=10)
cat_clf.fit(train_data)
```
✅ **No need for one-hot encoding or label encoding!**

---

## **6️⃣ Hyperparameter Tuning in CatBoost**
### **🔹 Key Hyperparameters**
| Parameter | Description |
|-----------|-------------|
| `iterations` | Number of boosting rounds |
| `learning_rate` | Step size (lower values prevent overfitting) |
| `depth` | Tree depth |
| `loss_function` | Loss function (`'Logloss'`, `'RMSE'`, etc.) |
| `eval_metric` | Metric used for evaluation (`'AUC'`, `'RMSE'`, etc.) |
| `early_stopping_rounds` | Stops training if no improvement is seen |

---

### **📌 Hyperparameter Tuning Example (Grid Search)**
```python
from sklearn.model_selection import GridSearchCV

cat_clf = CatBoostClassifier(verbose=0)

param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'iterations': [100, 300, 500]
}

grid_search = GridSearchCV(cat_clf, param_grid, scoring='accuracy', cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
```

---

## **7️⃣ Summary**
✔ **Handles categorical features without encoding.**  
✔ **Supports classification and regression tasks.**  
✔ **Faster training with ordered boosting.**  
✔ **Works efficiently on both CPU and GPU.**  

Would you like a **real-world dataset example (Titanic, House Prices)?** 🚀

---

### 📌 **⭐ If you found this useful, consider giving this repo a star!**  
🔗 **[Follow me for more ML content!](https://github.com/your-github)**  

---

Now you can directly **copy-paste** this into your GitHub repository! Let me know if you need any modifications. 🚀😊
