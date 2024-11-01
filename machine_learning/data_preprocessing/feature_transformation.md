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
