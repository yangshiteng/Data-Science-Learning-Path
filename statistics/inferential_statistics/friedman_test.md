![image](https://github.com/user-attachments/assets/ed29313a-768a-4298-bcd9-9387d3b68101)

![image](https://github.com/user-attachments/assets/b008f353-d108-4b90-9ff3-e62108067eb4)

![image](https://github.com/user-attachments/assets/d6e336ec-872e-46e3-99f5-c44456225d2f)

![image](https://github.com/user-attachments/assets/e5a638be-9827-4258-9767-bc2441332b22)

![image](https://github.com/user-attachments/assets/062e5770-4813-431d-96c0-ea464af420c5)

# Python Implementation for the Friedman Test

```python
import numpy as np
from scipy import stats

# Data for three different diets across three participants
diet_1 = np.array([5, 3, 6])
diet_2 = np.array([3, 2, 5])
diet_3 = np.array([2, 1, 4])

# Perform the Friedman Test
statistic, p_value = stats.friedmanchisquare(diet_1, diet_2, diet_3)

print(f'Friedman Test Statistic: {statistic}')
print(f'P-Value: {p_value}')

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference among the diets.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference among the diets.")

```
