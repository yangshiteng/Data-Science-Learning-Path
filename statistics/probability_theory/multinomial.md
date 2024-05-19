![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/da91cb76-8981-4c95-b35f-7c77bb4ffdfa)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7d435ee0-a61d-4fcb-a440-0bc3bdb953db)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a00e90ad-3a5d-4f70-b9f3-6b3feeb1b1b8)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
n = 10  # number of trials
p = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]  # probability vector for each outcome

# Create a multinomial distribution
multinomial_dist = stats.multinomial(n, p)

# Generate random samples
samples = multinomial_dist.rvs(size=1000)

# Calculate mean and variance
mean = multinomial_dist.mean()
variance = multinomial_dist.var()

print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Plot the PMF for the first outcome
x = np.arange(0, n+1)
pmf = multinomial_dist.pmf(np.column_stack([x, n-x, np.zeros((len(x), 4))]))
plt.stem(x, pmf, basefmt=" ")
plt.title('Multinomial Distribution PMF (First Outcome)')
plt.xlabel('Number of Successes for First Outcome')
plt.ylabel('Probability')
plt.xticks(np.arange(0, n+1))
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6bfbc378-f58d-4eae-8bcc-8becf437dd9e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/04bad9d1-61c4-410c-b872-6d333ec82388)
