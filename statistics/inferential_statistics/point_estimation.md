![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9dd6263f-8c25-49a3-a57b-44be4b2991eb)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b254237a-0d35-4bdc-aa43-4fdcdc93948b)

```python
import numpy as np

# Sample data: Heights of 10 students (in cm)
sample_heights = np.array([160, 165, 170, 175, 168, 162, 180, 177, 169, 172])

# Calculate the sample mean (point estimate of the population mean)
sample_mean = np.mean(sample_heights)

# Calculate the sample variance (point estimate of the population variance)
sample_variance = np.var(sample_heights, ddof=1)  # ddof=1 for sample variance

print(f"Sample Heights: {sample_heights}")
print(f"Sample Mean (Point Estimate of Population Mean): {sample_mean:.2f} cm")
print(f"Sample Variance (Point Estimate of Population Variance): {sample_variance:.2f} cm^2")
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2dc28907-354b-493a-8272-7d32c82ffc97)

