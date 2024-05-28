![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/eb54ba50-6a61-4ae2-bc77-a5c732d16847)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6cda99e4-71dd-4613-9b0d-34afd7ce842a)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/89bc40f3-d84d-42b8-aeb4-c3e892d086f3)

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
population_mean = 3.5
population_variance = 2.92
n = 30  # Sample size
num_simulations = 1000

# Simulate rolling a die
sample_means = []
for _ in range(num_simulations):
    sample = np.random.randint(1, 7, n)
    sample_means.append(np.mean(sample))

# Plot the histogram of sample means
plt.hist(sample_means, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')

# Plot the normal distribution curve
from scipy.stats import norm
x = np.linspace(min(sample_means), max(sample_means), 100)
plt.plot(x, norm.pdf(x, population_mean, np.sqrt(population_variance/n)), 'k', linewidth=2)
plt.title('Histogram of Sample Means (n=30) and Normal Distribution Curve')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/610734fb-9bab-41a1-9db0-d4dc3f40a12f)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/30b004b5-2983-4108-9248-19ce7561f80e)
