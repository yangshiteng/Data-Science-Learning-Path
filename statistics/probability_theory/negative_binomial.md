![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d306b67a-ed0e-46a0-8906-2820b42afc11)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e379fd5b-b586-4c25-b341-f4889292d96b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/933282a5-962b-4ac4-894c-0d0a2b47a684)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
r = 3  # number of successes
p = 0.5  # probability of success

# Create a negative binomial distribution
negbin_dist = stats.nbinom(r, p)

# Generate random samples
samples = negbin_dist.rvs(size=1000)

# Calculate mean and variance
mean = negbin_dist.mean()
variance = negbin_dist.var()

print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Plot the PMF
x = np.arange(0, 21)
pmf = negbin_dist.pmf(x)
plt.stem(x, pmf, basefmt=" ")
plt.title('Negative Binomial Distribution PMF')
plt.xlabel('Number of Failures')
plt.ylabel('Probability')
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ea4e841a-a30f-4e19-af67-897e44be5346)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/efbae10b-d3e9-46fb-b15d-35c1745066b3)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3531ab57-d368-466d-ac0a-e5a429e4da92)
