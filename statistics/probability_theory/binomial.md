![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a87fa333-409b-4dd7-bd40-9bfca9683c3e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d9b5b4b8-e35e-4e29-8028-20e5d79bb1ad)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/5cbef6f9-1012-4b55-b778-80409c0ef72c)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
n = 10  # number of trials
p = 0.5  # probability of success

# Create a binomial distribution
binom_dist = stats.binom(n, p)

# Generate random samples
samples = binom_dist.rvs(size=1000)

# Calculate mean and variance
mean = binom_dist.mean()
variance = binom_dist.var()

print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Plot the PMF
x = np.arange(0, n+1)
pmf = binom_dist.pmf(x)
plt.stem(x, pmf, basefmt=" ")
plt.title('Binomial Distribution PMF')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.xticks(np.arange(0, n+1))
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/10d380e1-f791-4fa0-b95f-cf3db5d8de66)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c39ebcfd-f40d-4e08-8c85-5609017b8154)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4b16d208-fd7d-449d-b4f0-400912d63597)
