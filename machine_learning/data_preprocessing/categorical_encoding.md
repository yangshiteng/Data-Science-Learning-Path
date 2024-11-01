![image](https://github.com/user-attachments/assets/01ed8f7d-7645-4704-8a7c-6c3e050117ee)

![image](https://github.com/user-attachments/assets/e772c944-76b8-4afc-8404-cfb8adf44413)

# Label Encoding
```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Sample data: Creating a DataFrame with a categorical feature
data = {'Brand': ['Apple', 'Samsung', 'Apple', 'Huawei', 'Samsung', 'Huawei']}
df = pd.DataFrame(data)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Brand' column
df['Brand_encoded'] = label_encoder.fit_transform(df['Brand'])

# Display the original and encoded data
print(df)
```
![image](https://github.com/user-attachments/assets/5745e154-9f95-48db-9903-484d116a5912)

# One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Sample data
data = {'Brand': ['Apple', 'Samsung', 'Apple', 'Huawei', 'Samsung', 'Huawei']}
df = pd.DataFrame(data)

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=True)  # Default behavior is to return a sparse matrix

# Fit and transform the 'Brand' column (reshape(-1, 1) because it needs to be 2D)
df_encoded_sparse = encoder.fit_transform(df[['Brand']])

# To convert the sparse matrix to a dense array, use toarray()
df_encoded = df_encoded_sparse.toarray()

# Convert the numpy array back to a DataFrame with appropriate column names
df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(['Brand']))

# Concatenate the original DataFrame with the new one-hot encoded DataFrame
df_final = pd.concat([df, df_encoded], axis=1)

# Display the result
print("One-Hot Encoded DataFrame:")
print(df_final)

```
![image](https://github.com/user-attachments/assets/40bf7b37-0312-458f-adf3-2e01a0330d96)

# Binary Encoding

```python
import pandas as pd
from category_encoders import BinaryEncoder

# Sample data
data = {
    'Brand': ['Apple', 'Samsung', 'Apple', 'Huawei', 'Samsung', 'Huawei', 'Xiaomi', 'Apple', 'Xiaomi', 'OnePlus']
}
df = pd.DataFrame(data)

# Initialize the BinaryEncoder
encoder = BinaryEncoder(cols=['Brand'])

# Fit and transform the data
df_encoded = encoder.fit_transform(df['Brand'])

# Display the original and encoded data
print("Original Data:")
print(df)
print("\nBinary Encoded Data:")
print(df_encoded)
```
![image](https://github.com/user-attachments/assets/36ff6b9a-6801-4d38-89e1-62b67bdc9889)

![image](https://github.com/user-attachments/assets/d87597da-48a6-4d3b-bdf9-84433cc16f55)
