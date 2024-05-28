![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e32fd557-6410-4d6d-8959-1207e9612e1c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d9bc8db7-9fe4-402c-ba35-1887e4ed9239)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0b72027d-278d-46ff-b6a5-61df060bbe2c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ee09fea3-d152-4c8c-bea4-f2f3d38b1102)

```python
import numpy as np
import scipy.stats as stats

# Given data
sample_mean = 170  # Sample mean height
sample_std = 10    # Sample standard deviation
n = 25             # Sample size
confidence_level = 0.95

# Degrees of freedom
df = n - 1

# Critical value from the t-distribution
t_critical = stats.t.ppf((1 + confidence_level) / 2, df)

# Standard error of the mean
SE = sample_std / np.sqrt(n)

# Margin of error
ME = t_critical * SE

# Confidence interval
confidence_interval = (sample_mean - ME, sample_mean + ME)

print(f"Sample mean: {sample_mean}")
print(f"Sample standard deviation: {sample_std}")
print(f"Sample size: {n}")
print(f"Confidence level: {confidence_level*100}%")
print(f"Critical value (t): {t_critical:.4f}")
print(f"Standard error: {SE:.4f}")
print(f"Margin of error: {ME:.4f}")
print(f"95% Confidence interval: {confidence_interval}")

```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/cd4dfa9f-0727-4942-9730-6c513ef8fd9c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c37ef0c3-0238-4e13-a0bc-882d1d51ccf8)
