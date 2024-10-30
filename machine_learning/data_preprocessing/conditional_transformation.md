![image](https://github.com/user-attachments/assets/91604283-74ab-457f-b36f-c3d82029d946)

![image](https://github.com/user-attachments/assets/bb3634ce-f76b-4078-8dfe-b8a8ce1c3d6b)

![image](https://github.com/user-attachments/assets/1238c8ba-eb84-449a-82dd-5fb0b750e8f9)

```python
import pandas as pd
import numpy as np

# Create sample DataFrame
data = {
    'Age': [22, 55, 13, 42, 65, 23],
    'Income': [50000, 120000, None, 75000, None, 45000],
    'Employment': ['Student', 'Professional', 'Student', 'Professional', 'Retired', 'Student']
}

df = pd.DataFrame(data)

# Conditional transformation: Binning ages into groups
df['Age Group'] = pd.cut(df['Age'], bins=[0, 20, 40, 60, 100], labels=['0-20', '21-40', '41-60', '61+'])

# Conditional transformation: Imputing missing Income based on Employment type
df.loc[df['Employment'] == 'Student', 'Income'] = df.loc[df['Employment'] == 'Student', 'Income'].fillna(20000)
df.loc[df['Employment'] == 'Professional', 'Income'] = df.loc[df['Employment'] == 'Professional', 'Income'].fillna(80000)
df.loc[df['Employment'] == 'Retired', 'Income'] = df.loc[df['Employment'] == 'Retired', 'Income'].fillna(40000)

print(df)

```
