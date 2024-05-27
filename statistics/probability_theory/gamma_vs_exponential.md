![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6aa1f504-072a-42be-9f76-5b42890ea206)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/1ea6412f-9a2f-42dc-9a6f-88fba9b27afc)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/55938da3-d1be-41f0-8326-982c6b767e77)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b2efbfe9-ad5b-4bdc-ab09-2527d3fec5c1)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
lambda_exp = 2  # rate parameter

# Create an exponential distribution
exp_dist = stats.expon(scale=1/lambda_exp)

# Generate random samples
samples_exp = exp_dist.rvs(size=1000)

# Plot the PDF
x_exp = np.linspace(0, 3, 1000)
pdf_exp = exp_dist.pdf(x_exp)
plt.plot(x_exp, pdf_exp, label='Exponential PDF')

# Plot the histogram of the samples
plt.hist(samples_exp, bins=30, density=True, alpha=0.6, color='g', label='Histogram of samples')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Exponential Distribution PDF')
plt.grid(True)
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/57433cca-f961-4e67-b2f2-5c6c0b1e0fc4)

```python
# Parameters
k = 3  # shape parameter
lambda_gamma = 2  # rate parameter

# Create a gamma distribution
gamma_dist = stats.gamma(a=k, scale=1/lambda_gamma)

# Generate random samples
samples_gamma = gamma_dist.rvs(size=1000)

# Plot the PDF
x_gamma = np.linspace(0, 5, 1000)
pdf_gamma = gamma_dist.pdf(x_gamma)
plt.plot(x_gamma, pdf_gamma, label='Gamma PDF')

# Plot the histogram of the samples
plt.hist(samples_gamma, bins=30, density=True, alpha=0.6, color='g', label='Histogram of samples')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Gamma Distribution PDF')
plt.grid(True)
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b1f1248e-abf6-44c4-bbeb-e93406eff9f4)


