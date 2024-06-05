![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e7c84466-69ed-4b2c-aa31-058a9a729089)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a7254863-7645-49c3-bfbb-38d9b9dba882)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6432d984-903d-4bfe-a95d-48cd781f9544)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2244b4df-e7c7-4ccb-9688-3904f9f3c385)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7e7faeca-4fb5-4145-91c5-1d0497653d38)

```python
import numpy as np
import scipy.stats as stats

# Sample data: Heights of 25 students (in cm)
sample_heights = np.array([160, 165, 170, 175, 168, 162, 180, 177, 169, 172, 160, 165, 170, 175, 168,
                           162, 180, 177, 169, 172, 160, 165, 170, 175, 168])

# Calculate the sample mean and sample standard deviation
sample_mean = np.mean(sample_heights)
sample_std_dev = np.std(sample_heights, ddof=1)  # ddof=1 for sample standard deviation
n = len(sample_heights)  # Sample size

# Select the confidence level
confidence_level = 0.95

# Find the critical value from the t-distribution
degrees_of_freedom = n - 1
t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

# Calculate the margin of error
standard_error = sample_std_dev / np.sqrt(n)
margin_of_error = t_critical * standard_error

# Construct the confidence interval
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print(f"Sample Mean: {sample_mean:.2f} cm")
print(f"Sample Standard Deviation: {sample_std_dev:.2f} cm")
print(f"Sample Size: {n}")
print(f"Confidence Level: {confidence_level*100}%")
print(f"Critical Value (t): {t_critical:.4f}")
print(f"Standard Error: {standard_error:.2f} cm")
print(f"Margin of Error: {margin_of_error:.2f} cm")
print(f"95% Confidence Interval: {confidence_interval}")
***
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/abe447fd-b425-45a8-989b-cf5c3ea2e4f4)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d4657f37-2341-4c1a-a366-1c94daf71d9f)






