![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/1b860131-139b-476f-8c7e-49e21bbee8d4)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/10f48a2e-b0bd-40e1-b0b1-eade5c118f36)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4b1774f0-a9de-4e7f-88d9-33bf15bea877)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
mu = 0  # mean
sigma = 1  # standard deviation

# Create a normal distribution
normal_dist = stats.norm(mu, sigma)

# Generate random samples
samples = normal_dist.rvs(size=1000)

# Calculate mean and variance
mean = np.mean(samples)
variance = np.var(samples)

print(f"Sample Mean: {mean}")
print(f"Sample Variance: {variance}")

# Plot the PDF
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
pdf = normal_dist.pdf(x)
plt.plot(x, pdf, label='PDF')

# Plot the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Histogram of samples')

# Plot the CDF
cdf = normal_dist.cdf(x)
plt.figure()
plt.plot(x, cdf, label='CDF')

plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4483f5f2-fb28-493e-9e97-b3d6113bcb5d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/fec6474b-5ede-45d4-92ee-01efebefe932)




