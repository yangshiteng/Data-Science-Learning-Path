![image](https://github.com/user-attachments/assets/9e6aa2a1-b384-422a-a4ed-1078ee11139f)

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
```

![image](https://github.com/user-attachments/assets/137c3c11-fdbe-4547-963d-14f79d95d35a)

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)
```

![image](https://github.com/user-attachments/assets/7d51fabc-9818-4bba-995a-14043c0d16ca)

```python
from sklearn.preprocessing import Normalizer
normalizer = Normalizer(norm='l2')
data_normalized = normalizer.fit_transform(data)
```

![image](https://github.com/user-attachments/assets/650fe9d4-0c66-404c-ada3-361eeeb48979)

```python
import pandas as pd
age_data = pd.DataFrame({'age': [22, 25, 37, 59, 45, 18, 33]})
age_data['age_binned'] = pd.cut(age_data['age'], bins=[0, 20, 40, 60], labels=["0-20", "21-40", "41-60"])
print(age_data)
```

![image](https://github.com/user-attachments/assets/098ae65c-0cb7-427a-b5ba-f659e2868b66)

```python
import numpy as np
sales_data = np.array([200, 300, 400, 600, 1000])
sales_log = np.log(sales_data)
```

![image](https://github.com/user-attachments/assets/bf33249c-6f85-4439-a919-1b5f788d4dfe)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

# Sample data
data = np.array([[100, 0.1],
                 [200, 0.15],
                 [300, 0.2],
                 [400, 0.25],
                 [500, 0.3]])

# Applying Min-Max Scaling
scaler_min_max = MinMaxScaler()
data_min_max_scaled = scaler_min_max.fit_transform(data)

# Applying Standard Scaling
scaler_std = StandardScaler()
data_std_scaled = scaler_std.fit_transform(data)

# Applying L2 Normalization
normalizer = Normalizer(norm='l2')
data_normalized = normalizer.fit_transform(data)

# Convert to DataFrame for better readability
df_transformed = pd.DataFrame({
    'Original Feature 1': data[:, 0],
    'Min-Max Scaled Feature 1': data_min_max_scaled[:, 0],
    'Standard Scaled Feature 1': data_std_scaled[:, 0],
    'L2 Normalized Feature 1': data_normalized[:, 0]
})

print(df_transformed)
```

![image](https://github.com/user-attachments/assets/ad23b1a8-7d1f-4ff1-a83a-e5355aeb39ed)
