# Introduction

![image](https://github.com/user-attachments/assets/f74c4813-23f2-46c5-ac69-fe2e62f74184)

![image](https://github.com/user-attachments/assets/ab3e12e7-f84b-4b4c-b7b3-c07d7f1d9b08)

![image](https://github.com/user-attachments/assets/5cd0ce0e-77ec-403c-a911-cdd5e4b47fe1)

# KNN Imputation

![image](https://github.com/user-attachments/assets/9a8d1662-b3c5-46f8-b2e4-0d2cdfc0323c)

![image](https://github.com/user-attachments/assets/c41df565-1116-4244-ada5-bcbb926f37b1)

```python
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Sample data: rows represent samples and columns represent features
data = np.array([
    [1, 2, np.nan],  # The third feature is missing in the first sample
    [4, 6, 5],
    [7, np.nan, 9],  # The second feature is missing in the third sample
    [10, 5, 7]
])

# Convert the array to a DataFrame for better visualization
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3'])

print("Original Data:")
print(df)

# Create the KNN imputer instance, assuming 2 nearest neighbors
imputer = KNNImputer(n_neighbors=2)

# Perform the imputation
imputed_data = imputer.fit_transform(data)

# Convert imputed data back to a DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=['Feature1', 'Feature2', 'Feature3'])

print("\nImputed Data:")
print(imputed_df)

```

# Regression Imputation

![image](https://github.com/user-attachments/assets/f0193b17-f6ed-46cd-a09d-664d0da7aebe)

![image](https://github.com/user-attachments/assets/1fd34320-2c25-4b12-a95a-5b13069b0f3a)

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Sample data with missing values
data = {
    'Feature1': [1, 2, 3, 4, np.nan],
    'Feature2': [2, 4, np.nan, 8, 10],
    'Feature3': [5, np.nan, 15, 20, 25]
}

df = pd.DataFrame(data)

# Assume 'Feature3' is the target/response variable, and we will use 'Feature1' and 'Feature2' as predictors
# Preparing training data (dropping rows where the target is NaN)
train_df = df.dropna(subset=['Feature3'])

# Build regression model
model = LinearRegression()
model.fit(train_df[['Feature1', 'Feature2']], train_df['Feature3'])

# Predict missing values in 'Feature3'
df['Feature3_Predicted'] = model.predict(df[['Feature1', 'Feature2']])

# Fill in missing 'Feature3' values using the predictions
df['Feature3'].fillna(df['Feature3_Predicted'], inplace=True)

# Clean up if desired
df.drop(columns=['Feature3_Predicted'], inplace=True)

print(df)
```













