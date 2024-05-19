![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f89bec48-353f-4f71-9545-8f5822013e9b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a1ce6a45-f788-4954-9e6a-759d9c1ce829)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3ce76d07-36d7-4bf5-acea-56776cb62f1a)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
N = 1000  # total population size
K = 100  # number of successes in the population
n = 10  # sample size

# Create a hypergeometric distribution
hypergeom_dist = stats.hypergeom(N, K, n)

# Generate random samples
samples = hypergeom_dist.rvs(size=1000)

# Calculate mean and variance
mean = hypergeom_dist.mean()
variance = hypergeom_dist.var()

print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Plot the PMF
x = np.arange(0, n+1)
pmf = hypergeom_dist.pmf(x)
plt.stem(x, pmf, basefmt=" ")
plt.title('Hypergeometric Distribution PMF')
plt.xlabel('Number of Successes in Sample')
plt.ylabel('Probability')
plt.xticks(np.arange(0, n+1))
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/754e8364-59c9-4ad0-b855-db752a17399e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d307e9a6-0779-47a8-abe2-a03f767a498e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c01a6e5e-da0f-456a-a0ef-41f67227fcfb)
