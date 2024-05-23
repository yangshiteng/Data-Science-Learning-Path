![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e0e13140-69ad-4e9a-a5bc-02ec898f5152)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/517cd956-a837-4a40-b2a8-f5fb273c1aad)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/815c2a42-02d0-4904-879e-7aa03ed1bc35)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
k = 5  # shape parameter
lambda_ = 3  # rate parameter

# Create a gamma distribution
gamma_dist = stats.gamma(a=k, scale=1/lambda_)

# Generate random samples
samples = gamma_dist.rvs(size=1000)

# Calculate mean and variance
mean = np.mean(samples)
variance = np.var(samples)

print(f"Sample Mean: {mean}")
print(f"Sample Variance: {variance}")

# Plot the PDF
x = np.linspace(0, 5, 1000)
pdf = gamma_dist.pdf(x)
plt.plot(x, pdf, label='PDF')

# Plot the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Histogram of samples')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Gamma Distribution PDF')
plt.grid(True)
plt.show()

# Plot the CDF
cdf = gamma_dist.cdf(x)
plt.figure()
plt.plot(x, cdf, label='CDF')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.title('Gamma Distribution CDF')
plt.grid(True)
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4ce25a9c-99f1-4b65-85e9-745ec648c023)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/8349cc2b-817e-460a-a328-16a2b33ee01e)


