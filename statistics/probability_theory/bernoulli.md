![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2ac2ee1f-148c-4890-adde-add277aadc7d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/8f65b509-13c2-4f2a-b385-141d33c005a1)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/883fbb44-d287-48a6-82e5-d3fb6e73cb64)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
p = 0.5  # probability of success

# Create a Bernoulli distribution
bernoulli_dist = stats.bernoulli(p)

# Generate random samples
samples = bernoulli_dist.rvs(size=1000)

# Calculate mean and variance
mean = bernoulli_dist.mean()
variance = bernoulli_dist.var()

print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Plot the PMF
x = [0, 1]
pmf = bernoulli_dist.pmf(x)
plt.stem(x, pmf)
plt.title('Bernoulli Distribution PMF')
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.xticks([0, 1])
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/40ba482b-205d-44ff-a73e-376a446b8265)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d71ac365-ccac-4948-be1a-163f98695dd9)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e5322790-6fb0-4229-b3a9-b68f44d598ff)
