![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a9992bb8-5da9-4654-a8e5-b4c37354113d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/83697319-d90b-490b-9686-696dbc091d53)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/48b1e247-f6d3-4d43-a77a-9e4d2b33e2b6)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
df = 9  # degrees of freedom

# Create a t-distribution
t_dist = stats.t(df)

# Generate random samples
samples = t_dist.rvs(size=1000)

# Calculate mean and variance
mean = np.mean(samples)
variance = np.var(samples)

print(f"Sample Mean: {mean}")
print(f"Sample Variance: {variance}")

# Plot the PDF
x = np.linspace(-4, 4, 1000)
pdf = t_dist.pdf(x)
plt.plot(x, pdf, label='PDF')

# Plot the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Histogram of samples')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Student\'s t-Distribution PDF')
plt.grid(True)
plt.show()

# Plot the CDF
cdf = t_dist.cdf(x)
plt.figure()
plt.plot(x, cdf, label='CDF')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.title('Student\'s t-Distribution CDF')
plt.grid(True)
plt.show()

```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/08adb5f6-30bb-448d-9492-9843b4729dcc)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/482f6561-ebfc-4abc-9772-da4dce3099b7)











