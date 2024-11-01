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

![image](https://github.com/user-attachments/assets/d87597da-48a6-4d3b-bdf9-84433cc16f55)
