![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d18e6012-348a-4eac-934c-4a49e2ac8350)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/cef239d1-a235-4189-b95c-c70e4a739710)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/463430bb-7f2a-4268-b757-bff5c24c0558)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/cc7278f5-a6de-49fd-89ce-6c81333749e0)

```python
import numpy as np
import scipy.stats as stats

# Sample data: Heights of 25 students (in cm)
sample_heights = np.array([160, 165, 170, 175, 168, 162, 180, 177, 169, 172, 160, 165, 170, 175, 168,
                           162, 180, 177, 169, 172, 160, 165, 170, 175, 168])

# Known population standard deviation
sigma = 6

# Calculate the sample mean
sample_mean = np.mean(sample_heights)
n = len(sample_heights)  # Sample size

# Select the confidence level
confidence_level = 0.95

# Find the critical value from the standard normal distribution
z_critical = stats.norm.ppf((1 + confidence_level) / 2)

# Calculate the standard error
standard_error = sigma / np.sqrt(n)

# Calculate the margin of error
margin_of_error = z_critical * standard_error

# Construct the confidence interval
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print(f"Sample Mean: {sample_mean:.2f} cm")
print(f"Known Population Standard Deviation: {sigma} cm")
print(f"Sample Size: {n}")
print(f"Confidence Level: {confidence_level*100}%")
print(f"Critical Value (z): {z_critical:.4f}")
print(f"Standard Error: {standard_error:.2f} cm")
print(f"Margin of Error: {margin_of_error:.2f} cm")
print(f"95% Confidence Interval: {confidence_interval}")
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/246e236a-e6d5-48f9-b438-fcde887e751d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0fc01093-85a0-4810-8bb6-ee4ceaf71d79)


