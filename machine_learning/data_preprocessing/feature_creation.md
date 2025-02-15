![image](https://github.com/user-attachments/assets/7ac675d1-65b4-4175-9d2d-5b726b0e75e4)

![image](https://github.com/user-attachments/assets/196d7307-d612-46fd-9d27-44513d83e0ee)

![image](https://github.com/user-attachments/assets/9843af45-d451-41e7-97e0-1847dc6d6274)

![image](https://github.com/user-attachments/assets/52ef0183-08f8-48be-ae47-3f1b167490d2)

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Sample data
data = {
    'age': [25, 30, 35, 40],
    'salary': [50000, 60000, 70000, 80000]
}
df = pd.DataFrame(data)

# Creating polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['age', 'salary']])

# For scikit-learn 0.21 and later
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['age', 'salary']))

print(poly_df)
```
![image](https://github.com/user-attachments/assets/3e19df54-70c4-4d7e-a4f5-b43936c1731d)

