![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/09d33ac2-0803-4099-8f7a-e22b80cb9d16)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a4a19dcf-9631-4022-9f31-c9e7ec17ace1)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9ecdf19e-a7cd-40eb-96ac-4674d1470e47)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
alpha = 2  # shape parameter
beta = 5  # shape parameter

# Create a beta distribution
beta_dist = stats.beta(alpha, beta)

# Generate random samples
samples = beta_dist.rvs(size=1000)

# Calculate mean and variance
mean = np.mean(samples)
variance = np.var(samples)

print(f"Sample Mean: {mean}")
print(f"Sample Variance: {variance}")

# Plot the PDF
x = np.linspace(0, 1, 1000)
pdf = beta_dist.pdf(x)
plt.plot(x, pdf, label='PDF')

# Plot the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Histogram of samples')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Beta Distribution PDF')
plt.grid(True)
plt.show()

# Plot the CDF
cdf = beta_dist.cdf(x)
plt.figure()
plt.plot(x, cdf, label='CDF')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.title('Beta Distribution CDF')
plt.grid(True)
plt.show()

```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6e58b717-4c0d-4c34-92b5-479740e74860)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ee0dd4d2-6f3e-4223-a847-4b846d89e468)



