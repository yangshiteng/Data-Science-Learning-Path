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

![image](https://github.com/user-attachments/assets/32c4c9e6-75dc-4d84-9ea3-16bca0436461)

![image](https://github.com/user-attachments/assets/7c9c768e-ab70-4d5c-8877-3e08a1727100)

```python
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Sample data with missing values in neighbors
data = np.array([
    [1, 2, np.nan],  # The third feature is missing in the first sample
    [4, np.nan, 5],  # The second feature is missing in the second sample
    [7, np.nan, 9],  # The second feature is missing in the third sample
    [10, 5, 7]
])

# Convert the array to a DataFrame for better visualization
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3'])

print("Original Data:")
print(df)

# Create an iterative imputer instance
imputer = IterativeImputer(max_iter=10, random_state=0)

# Perform the imputation
imputed_data = imputer.fit_transform(data)

# Convert imputed data back to a DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=['Feature1', 'Feature2', 'Feature3'])

print("\nImputed Data:")
print(imputed_df)

```












