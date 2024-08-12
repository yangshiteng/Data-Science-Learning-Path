![image](https://github.com/user-attachments/assets/221a866e-f5cc-4a94-be3d-1d2a6277a4a0)

![image](https://github.com/user-attachments/assets/48b3dafc-400f-4f1e-b74e-d5f4bf1b9c2e)

![image](https://github.com/user-attachments/assets/ecec1478-249b-487a-a7d8-b929d6c95ef9)

![image](https://github.com/user-attachments/assets/0640172f-8036-4323-a7fb-d380eb67a6f1)

![image](https://github.com/user-attachments/assets/8dc10309-d4ab-48b3-b088-5801aed8c5a8)

![image](https://github.com/user-attachments/assets/7c61fb6c-cbf4-4689-9533-6524f05d55aa)

# Python Implementation for Kruskal-Wallis H Test

```python
import numpy as np
from scipy import stats

# Data from three different diets
diet_A = np.array([3, 2, 1, 5, 4])
diet_B = np.array([6, 5, 7, 4])
diet_C = np.array([8, 9, 6, 7, 8, 10])

# Perform the Kruskal-Wallis H Test
h_statistic, p_value = stats.kruskal(diet_A, diet_B, diet_C)

print(f'Kruskal-Wallis H Statistic: {h_statistic}')
print(f'P-Value: {p_value}')

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the diets.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the diets.")
```
