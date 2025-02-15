![image](https://github.com/user-attachments/assets/e1ab8e79-b87d-4c6c-9fe8-8701c8c98f19)

![image](https://github.com/user-attachments/assets/2c84fc26-ecbf-447d-8d3c-1e0330cee821)

![image](https://github.com/user-attachments/assets/77327abe-e5fa-484e-9068-a6c4effc846c)

![image](https://github.com/user-attachments/assets/61e11417-dc2c-4e16-afd4-098d97575b33)

```python
import pandas as pd

# Sample data
data = {
    'age': [25, 30, 35, 40],
    'salary': [50000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)

# Creating an interaction feature
df['age_salary_interaction'] = df['age'] * df['salary']

print(df)
```
![image](https://github.com/user-attachments/assets/ab15a7e4-ee07-43e9-8f93-f045ef81d9e4)
