![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b435cae3-78fe-45fd-bb67-912c677fc8c8)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9aed0873-c430-4399-8647-e50b43146b6c)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
lambda_ = 1  # rate parameter

# Create an exponential distribution
exponential_dist = stats.expon(scale=1/lambda_)

# Generate random samples
samples = exponential_dist.rvs(size=1000)

# Calculate mean and variance
mean = np.mean(samples)
variance = np.var(samples)

print(f"Sample Mean: {mean}")
print(f"Sample Variance: {variance}")

# Plot the PDF
x = np.linspace(0, 10, 1000)
pdf = exponential_dist.pdf(x)
plt.plot(x, pdf, label='PDF')

# Plot the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Histogram of samples')

# Plot the CDF
cdf = exponential_dist.cdf(x)
plt.figure()
plt.plot(x, cdf, label='CDF')

plt.title('Exponential Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/da632e11-8cd9-4735-98c4-8430af84258d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/47bd7233-4fc7-4ccd-85b1-5f6a00b12027)
