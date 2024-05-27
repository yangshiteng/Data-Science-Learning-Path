![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ab937906-0fb0-4d2f-bb51-c0307d2628bc)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3060e001-a753-4463-a260-4e27cbe65bd2)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/37eb326d-23f2-441d-ab0b-1eb8cb53d397)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
k = 5  # degrees of freedom

# Create a chi-square distribution
chi2_dist = stats.chi2(df=k)

# Generate random samples
samples = chi2_dist.rvs(size=1000)

# Calculate mean and variance
mean = np.mean(samples)
variance = np.var(samples)

print(f"Sample Mean: {mean}")
print(f"Sample Variance: {variance}")

# Plot the PDF
x = np.linspace(0, 20, 1000)
pdf = chi2_dist.pdf(x)
plt.plot(x, pdf, label='PDF')

# Plot the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Histogram of samples')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Chi-Square Distribution PDF')
plt.grid(True)
plt.show()

# Plot the CDF
cdf = chi2_dist.cdf(x)
plt.figure()
plt.plot(x, cdf, label='CDF')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.title('Chi-Square Distribution CDF')
plt.grid(True)
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b7a1c72e-459f-4b7e-bac1-5e81a43ff1be)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/70c1385a-ba5f-4f85-8775-c2597d3871c4)




