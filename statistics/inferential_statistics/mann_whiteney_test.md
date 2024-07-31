![image](https://github.com/user-attachments/assets/3955b762-6a05-443b-bb05-1a7c7cc06a56)

![image](https://github.com/user-attachments/assets/ea02b628-dbbf-4cef-b361-c16c6c323e40)

![image](https://github.com/user-attachments/assets/f2da96fe-5a0d-4b03-ae55-e7b8f7f7f26d)

![image](https://github.com/user-attachments/assets/b3f00d87-3710-47f8-a878-e1c4e8aba036)

![image](https://github.com/user-attachments/assets/0052f904-cdd7-44ae-9a72-a5f16f255500)

![image](https://github.com/user-attachments/assets/236cd1fc-d608-44a2-83fb-b64c93d96bbf)

![image](https://github.com/user-attachments/assets/f0b0e921-25d0-4a46-8d4f-3cf678b4c4c2)

```python
import numpy as np
from scipy import stats

# Data from two different teaching methods
data_A = np.array([85, 78, 90, 87, 84])
data_B = np.array([80, 75, 85, 88, 83])

# Perform the Mann-Whitney U Test
u_statistic, p_value = stats.mannwhitneyu(data_A, data_B, alternative='two-sided')

print(f'Mann-Whitney U Statistic: {u_statistic}')
print(f'P-Value: {p_value}')

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the two teaching methods.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the two teaching methods.")
```
