![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/cf10f552-4329-48ee-acb2-93f2e1c3ec48)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/28a7f686-d499-4f51-8ff8-656738725ad8)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/5c9e174b-9e0d-419a-9c5e-daadc8fcaa93)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
lambda_ = 5  # average rate of occurrence

# Create a Poisson distribution
poisson_dist = stats.poisson(lambda_)

# Generate random samples
samples = poisson_dist.rvs(size=1000)

# Calculate mean and variance
mean = poisson_dist.mean()
variance = poisson_dist.var()

print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Plot the PMF
x = np.arange(0, 15)
pmf = poisson_dist.pmf(x)
plt.stem(x, pmf, basefmt=" ")
plt.title('Poisson Distribution PMF')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.xticks(np.arange(0, 15))
plt.show()
```

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f4df792e-ad1b-421f-abe4-d8032a968ec3)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/cb9ca75f-7abf-4cc2-bf1b-aa3335e42cfe)
