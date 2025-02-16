![image](https://github.com/user-attachments/assets/a0f54416-e146-478c-a9f7-2f13f0c0ecce)

![image](https://github.com/user-attachments/assets/148abeab-37e7-4868-90d7-c371e4e9152b)

![image](https://github.com/user-attachments/assets/1acf5735-a9d8-4e1c-9f06-24a4cd425ac5)

![image](https://github.com/user-attachments/assets/226f9b54-868d-4193-9a96-5eba943b9924)

![image](https://github.com/user-attachments/assets/89bc6a22-7cbd-42a0-8320-426d77435b82)

![image](https://github.com/user-attachments/assets/2fc341b9-17b9-49e2-9f7e-40e98c21b414)

![image](https://github.com/user-attachments/assets/568b6e09-c710-4cc3-a2d5-9056f27206a4)

![image](https://github.com/user-attachments/assets/b96c223f-3e62-4e73-8d66-feda635d578c)

```python
import numpy as np
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

# Generate training data (normal observations)
X_train = 0.3 * np.random.randn(100, 2)

# Generate test data (normal and abnormal observations)
X_test_inliers = 0.3 * np.random.randn(20, 2)
X_test_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X_test = np.vstack((X_test_inliers, X_test_outliers))

# Visualize the data
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c='green', label='Test Data')
plt.title("Synthetic Dataset")
plt.legend()
plt.show()

# Instantiate the One-Class SVM model
model = OneClassSVM(kernel="rbf", gamma=0.1, nu=0.05)

# Fit the model to the training data
model.fit(X_train)

# Predict the labels of the test data (+1 for inliers, -1 for outliers)
y_pred_test = model.predict(X_test)

# Separate inliers and outliers
X_test_inliers_pred = X_test[y_pred_test == 1]
X_test_outliers_pred = X_test[y_pred_test == -1]

# Plot the results
plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Training Data')
plt.scatter(X_test_inliers_pred[:, 0], X_test_inliers_pred[:, 1], c='green', label='Predicted Inliers')
plt.scatter(X_test_outliers_pred[:, 0], X_test_outliers_pred[:, 1], c='red', label='Predicted Outliers')
plt.title("One-Class SVM Results")
plt.legend()
plt.show()

```

![image](https://github.com/user-attachments/assets/c9b35649-5eed-444b-aeaf-7729adb5f833)

![image](https://github.com/user-attachments/assets/1ebedd65-ca8d-403b-b717-233511ddce94)

```python
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV

# Generate synthetic training data
np.random.seed(42)
X_train = 0.3 * np.random.randn(100, 2)

# Generate synthetic test data (for evaluation after grid search)
X_test_inliers = 0.3 * np.random.randn(20, 2)  # Inliers
X_test_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))  # Outliers
X_test = np.vstack((X_test_inliers, X_test_outliers))

# Create synthetic labels for evaluation (not used in GridSearchCV)
y_test = np.array([1] * 20 + [-1] * 20)  # 1 for inliers, -1 for outliers

# Custom scoring function
def one_class_svm_score(estimator, X):
    """
    Custom scoring function for One-Class SVM:
    Measures the fraction of training points predicted as inliers (+1).
    """
    y_pred = estimator.predict(X)
    return np.mean(y_pred == 1)

# Define parameter grid for Grid Search
param_grid = {
    'nu': [0.01, 0.05, 0.1],  # Controls the fraction of outliers
    'gamma': [0.01, 0.1, 1, 10]  # RBF kernel parameter
}

# Initialize One-Class SVM
ocsvm = OneClassSVM(kernel='rbf')

# Perform Grid Search
grid_search = GridSearchCV(
    estimator=ocsvm,
    param_grid=param_grid,
    scoring=one_class_svm_score,
    refit=True,
    cv=3  # Cross-validation on training data
)
grid_search.fit(X_train)

# Output the best parameters and best score
print("Best parameters found:")
print(grid_search.best_params_)
print("Best score on training data:", grid_search.best_score_)

# Use the best model for predictions on test data
best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)

# Separate inliers and outliers
X_test_inliers_pred = X_test[y_pred_test == 1]
X_test_outliers_pred = X_test[y_pred_test == -1]

# Visualize the results
import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], X_train[:, 1], color='blue', label='Training Data')
plt.scatter(X_test_inliers_pred[:, 0], X_test_inliers_pred[:, 1], color='green', label='Predicted Inliers')
plt.scatter(X_test_outliers_pred[:, 0], X_test_outliers_pred[:, 1], color='red', label='Predicted Outliers')
plt.legend()
plt.title("One-Class SVM with Grid Search Results")
plt.show()

```
![image](https://github.com/user-attachments/assets/d398bd1b-100e-4832-9e85-e40466ff0393)

![image](https://github.com/user-attachments/assets/650fce9d-5520-4962-a1f5-809bcc01a94f)

![image](https://github.com/user-attachments/assets/61af3fc3-82fb-4f56-ba93-639c550430dc)

![image](https://github.com/user-attachments/assets/0cc36720-ddbf-44f0-bde9-70779a041079)

# [Additional Code](https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/machine_learning/data_preprocessing/one_class_support_vector_machine_for_anomaly_detection.ipynb)





