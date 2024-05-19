![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/27e5f038-0529-4e94-8bd8-92d2bc6c8f5e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7bbb8f8f-929f-4d3d-8fd5-a3d0dffd32d4)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b309d1ff-7d0f-4bd6-9065-bbc38128d393)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
p = 0.5  # probability of success

# Create a geometric distribution
geo_dist = stats.geom(p)

# Generate random samples
samples = geo_dist.rvs(size=1000)

# Calculate mean and variance
mean = geo_dist.mean()
variance = geo_dist.var()

print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Plot the PMF
x = np.arange(1, 11)
pmf = geo_dist.pmf(x)
plt.stem(x, pmf, basefmt=" ")
plt.title('Geometric Distribution PMF')
plt.xlabel('Number of Trials')
plt.ylabel('Probability')
plt.xticks(np.arange(1, 11))
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/11a1ea07-76e4-4e8d-bcb1-e83d01f5ad75)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c245e4b2-2722-42e9-b3ce-59d512da7441)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/5edc3d6c-efde-4196-9f09-c2232ce12b60)
