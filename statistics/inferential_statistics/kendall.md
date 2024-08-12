![image](https://github.com/user-attachments/assets/630d98a0-75de-400a-8aa0-4a67cf530028)

![image](https://github.com/user-attachments/assets/2c181488-0b23-4a67-abe4-0756f996609e)

![image](https://github.com/user-attachments/assets/ffcffc4f-680f-4456-858b-30d4dd30a729)

![image](https://github.com/user-attachments/assets/b59cb870-9534-42ca-9163-b8a3dc9967c2)

# Python Implementation for Kendall’s Tau

```python
import numpy as np
from scipy import stats

# Data for hours studied and corresponding grades received
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
grades_received = np.array([55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

# Perform Kendall's Tau Test
tau, p_value = stats.kendalltau(hours_studied, grades_received)

print(f'Kendall’s Tau Coefficient: {tau:.3f}')
print(f'P-Value: {p_value:.3f}')

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant correlation between hours studied and grades received.")
else:
    print("Fail to reject the null hypothesis: There is no significant correlation between hours studied and grades received.")

```
