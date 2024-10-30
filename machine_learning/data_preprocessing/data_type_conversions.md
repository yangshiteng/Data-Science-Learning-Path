![image](https://github.com/user-attachments/assets/c3feea71-fece-48dc-88fd-10ec93e522e1)

![image](https://github.com/user-attachments/assets/01174383-c805-4073-ba33-b150a4f9915a)

![image](https://github.com/user-attachments/assets/8afc904b-4798-4250-9f4b-98fc477d34c0)

```python
import pandas as pd

# Create a sample DataFrame
data = {'ProductID': ['001', '002', '003'], 'Price': ['30.5', '45.2', '12.3'], 'InStock': [1, 0, 1]}
df = pd.DataFrame(data)

# Convert string to numeric
df['Price'] = pd.to_numeric(df['Price'])

# Convert integer to boolean
df['InStock'] = df['InStock'].astype(bool)

# Convert string to categorical
df['ProductID'] = df['ProductID'].astype('category')

print(df)
print(df.dtypes)
```
