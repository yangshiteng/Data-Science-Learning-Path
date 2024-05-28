![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/75cc2467-2743-48c7-a637-dd9d5cd93030)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f469141a-fce7-455c-adb8-41c9a4a4685d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/63a4e8fa-8dda-4fe6-b183-4ba3bc6c528d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/bb3ed8e8-2082-4fdf-9cb4-d123f216517a)

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 1000  # Number of rolls
num_simulations = 100  # Number of simulations

# Simulate rolling a die
sample_means = []
for _ in range(num_simulations):
    sample = np.random.randint(1, 7, n)
    sample_means.append(np.mean(sample))

# Plot the sample means
plt.plot(range(num_simulations), sample_means, marker='o')
plt.axhline(y=3.5, color='r', linestyle='-')
plt.title('Convergence of Sample Means to Population Mean (WLLN)')
plt.xlabel('Simulation Number')
plt.ylabel('Sample Mean')
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/cfe8485e-8f8a-432a-aebc-43c03a65324c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9af56a2a-da29-4af1-9bc8-201bb3608914)

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 1000  # Number of coin tosses

# Simulate coin tosses
coin_tosses = np.random.binomial(1, 0.5, n)
cumulative_means = np.cumsum(coin_tosses) / np.arange(1, n + 1)

# Plot the cumulative sample means
plt.plot(range(1, n + 1), cumulative_means, marker='o', markersize=2)
plt.axhline(y=0.5, color='r', linestyle='-')
plt.title('Convergence of Cumulative Sample Means to Population Mean (SLLN)')
plt.xlabel('Number of Tosses')
plt.ylabel('Cumulative Sample Mean')
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/33796ab4-0f6c-4ecb-bfda-52217610a119)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4d3e450e-1b9b-48bb-b3b5-6e8c662f1646)




