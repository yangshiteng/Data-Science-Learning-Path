![image](https://github.com/user-attachments/assets/3a2b72c3-1ea6-4fad-8d0b-34a7196d79ee)

![image](https://github.com/user-attachments/assets/90fa448b-b230-410b-9d81-a121f91f625b)

![image](https://github.com/user-attachments/assets/abe4c911-e455-47c2-94a5-c87eccfd7aaf)

# Python Implementation of the Shapiro-Wilk Test

```python
import numpy as np
from scipy import stats

# Example dataset: Response times in a cognitive test
response_times = np.array([12.5, 13.0, 12.7, 12.9, 13.1, 12.3, 11.8, 12.6, 13.0, 12.2, 
                           12.8, 12.9, 13.1, 13.2, 11.9, 12.0, 12.1, 12.4, 12.5, 12.7])

# Perform the Shapiro-Wilk Test
w_statistic, p_value = stats.shapiro(response_times)

print(f'Shapiro-Wilk Statistic: {w_statistic:.3f}')
print(f'P-Value: {p_value:.3f}')

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The data do not come from a normally distributed population.")
else:
    print("Fail to reject the null hypothesis: The data come from a normally distributed population.")
```
