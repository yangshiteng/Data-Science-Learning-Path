# Introduction

![image](https://github.com/user-attachments/assets/8c423c39-7701-4c48-892a-e9934b05ab3c)

![image](https://github.com/user-attachments/assets/5aa6c5c3-0f7d-4f61-b7bf-c9289fff198f)

```python
import pandas as pd
age_data = pd.DataFrame({'age': [22, 25, 37, 59, 45, 18, 33]})
age_data['age_binned'] = pd.cut(age_data['age'], bins=[0, 20, 40, 60], labels=["0-20", "21-40", "41-60"])
print(age_data)
```
![image](https://github.com/user-attachments/assets/fce2352e-fd70-423c-8f93-20c5e4844dfd)

# How Normalization and Standardization Help in Convergence

![image](https://github.com/user-attachments/assets/505d91cc-a231-4376-8250-4704d9d09f78)

![image](https://github.com/user-attachments/assets/8e6f9248-ba7a-4f9c-8879-310d17b58875)


