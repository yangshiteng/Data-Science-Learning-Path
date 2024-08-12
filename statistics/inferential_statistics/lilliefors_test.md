![image](https://github.com/user-attachments/assets/3a9d8e41-1669-4693-ac0f-92935f8869f9)

![image](https://github.com/user-attachments/assets/7fdbf22c-c8fe-4dd3-9f2e-e376a6f511a0)

![image](https://github.com/user-attachments/assets/dd98d07c-fa26-4027-962e-e0655cf171ec)

![image](https://github.com/user-attachments/assets/44092a02-c7ab-4f5a-8654-d41b940f1562)

![image](https://github.com/user-attachments/assets/15d9f2e7-aeba-4682-a9fc-962a4f625d86)

# Python Script for the Lilliefors Test

```python
import numpy as np
from statsmodels.stats.diagnostic import lilliefors

# Example data
data = np.array([70, 75, 80, 60, 85, 90, 65, 72, 68, 88])

# Perform the Lilliefors test
statistic, p_value = lilliefors(data, dist='norm')

# Output the results
print(f"Lilliefors Test Statistic: {statistic:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis: The data do not come from a normally distributed population.")
else:
    print("Fail to reject the null hypothesis: The data come from a normally distributed population.")

```
![image](https://github.com/user-attachments/assets/58adb33f-1d8e-4abb-b316-af674d0e65b0)
