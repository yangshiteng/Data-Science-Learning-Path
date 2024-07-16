# Introduction

![image](https://github.com/user-attachments/assets/3801d5ad-1ce7-409e-bb9b-971a22bbcd05)

![image](https://github.com/user-attachments/assets/a6e4756d-f566-44a2-8e73-ddf8ae350cbd)

![image](https://github.com/user-attachments/assets/4d8f0a51-61b2-4f52-b628-27a01967cd8e)

![image](https://github.com/user-attachments/assets/faa9b0a5-12cd-4b81-9249-a62afeb710d2)

![image](https://github.com/user-attachments/assets/ca21997b-0960-4ca4-a922-85cdab7ceafc)

![image](https://github.com/user-attachments/assets/eb9e5401-48e1-455a-b4e1-af15bdd4325c)

![image](https://github.com/user-attachments/assets/fe421e1a-78f7-460f-9acc-2a941f72ef4a)

# Python Implementation

## One-Sample K-S Test Implementation

```python
import numpy as np
from scipy import stats

# Sample data
data = [-1.2, -0.5, 0.1, 0.2, 0.4, 0.8, 1.1, 1.3, 1.5, 1.8]

# Perform the K-S test
ks_statistic, p_value = stats.kstest(data, 'norm', args=(0, 1))

print(f'K-S Statistic: {ks_statistic}')
print(f'P-Value: {p_value}')

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The sample does not come from the specified normal distribution.")
else:
    print("Fail to reject the null hypothesis: The sample comes from the specified normal distribution.")

```
