![image](https://github.com/user-attachments/assets/05ceee3b-347f-4116-8d1b-d5c54f8827f5)

![image](https://github.com/user-attachments/assets/205d0650-c366-47cd-b3ed-93547107e677)

![image](https://github.com/user-attachments/assets/97943c88-eea5-4030-ad8a-552e3a2350c7)

![image](https://github.com/user-attachments/assets/906e30fb-4935-44a3-8ee3-3ec38bce6e1b)

# Python Implementation for Wilcoxon Signed-Rank Test

```python
import numpy as np
from scipy import stats

# Data before and after the intervention
scores_before = np.array([82, 90, 76, 85, 88])
scores_after = np.array([89, 92, 78, 88, 91])

# Perform the Wilcoxon Signed-Rank Test
statistic, p_value = stats.wilcoxon(scores_before, scores_after)

print(f'Wilcoxon Signed-Rank Statistic: {statistic}')
print(f'P-Value: {p_value}')

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference between the before and after scores.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the before and after scores.")

```
