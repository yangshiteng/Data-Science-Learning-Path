![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/891057c4-4935-41cb-9e8e-b3f4dc58ae32)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/76f86dbe-bd7a-401c-8a30-62c7dd23a4a8)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
a = 0  # minimum value
b = 1  # maximum value

# Create a uniform distribution
uniform_dist = stats.uniform(a, b-a)

# Generate random samples
samples = uniform_dist.rvs(size=1000)

# Calculate mean and variance
mean = np.mean(samples)
variance = np.var(samples)

print(f"Sample Mean: {mean}")
print(f"Sample Variance: {variance}")

# Plot the PDF
x = np.linspace(a, b, 1000)
pdf = uniform_dist.pdf(x)
plt.plot(x, pdf, label='PDF')

# Plot the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Histogram of samples')

# Plot the CDF
cdf = uniform_dist.cdf(x)
plt.figure()
plt.plot(x, cdf, label='CDF')

plt.title('Uniform Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2590b5a7-3e61-48d2-bdca-1e2e215a1ce8)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/059ccc42-a861-49ee-b90b-cf7992a264de)













