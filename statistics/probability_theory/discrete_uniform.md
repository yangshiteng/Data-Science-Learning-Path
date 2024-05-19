![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/aa7e178a-8484-4934-9a9c-8dce53886380)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/8aa34755-ff9b-40d3-b702-dc93c43e451e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/84317a4a-e1c7-4909-a243-b25eebd15c74)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
a = 1  # minimum value
b = 6  # maximum value

# Create a discrete uniform distribution
discrete_uniform_dist = stats.randint(a, b+1)

# Generate random samples
samples = discrete_uniform_dist.rvs(size=1000)

# Calculate mean and variance
mean = discrete_uniform_dist.mean()
variance = discrete_uniform_dist.var()

print(f"Mean: {mean}")
print(f"Variance: {variance}")

# Plot the PMF
x = np.arange(a, b+1)
pmf = discrete_uniform_dist.pmf(x)
plt.stem(x, pmf, basefmt=" ")
plt.title('Discrete Uniform Distribution PMF')
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.xticks(np.arange(a, b+1))
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/889b2fb9-7aa1-48c0-b557-345326658e72)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/12e6e567-00c1-445b-a311-a36c3e6dd123)
