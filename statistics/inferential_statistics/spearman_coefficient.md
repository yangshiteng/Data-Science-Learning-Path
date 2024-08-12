![image](https://github.com/user-attachments/assets/d8c99234-d783-4c32-98b5-48ded48e82c2)

![image](https://github.com/user-attachments/assets/f25858bb-1fcc-455e-a7bd-eec4e9d05104)

![image](https://github.com/user-attachments/assets/83d47158-b885-4ded-b96c-8553e22ec2bf)

![image](https://github.com/user-attachments/assets/0b2de5f7-23db-4f44-b724-26763bbd29e0)

# Python Implementation for Spearman’s Rank Correlation Coefficient

```python
import numpy as np
from scipy import stats

# Data for hours studied and corresponding test scores
hours_studied = np.array([1, 2, 3, 4, 5])
test_scores = np.array([10, 20, 30, 25, 40])

# Perform Spearman's Rank Correlation Test
correlation, p_value = stats.spearmanr(hours_studied, test_scores)

print(f'Spearman\'s Rank Correlation Coefficient: {correlation:.3f}')
print(f'P-Value: {p_value:.3f}')

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant correlation between hours studied and test scores.")
else:
    print("Fail to reject the null hypothesis: There is no significant correlation between hours studied and test scores.")

```
