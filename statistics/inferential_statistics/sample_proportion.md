![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f8d5dab8-c656-4ef2-95e7-20bc8a2f6e2b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/29b14303-0d19-439d-afc9-2886ebcb24ca)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/be08d1cb-8a64-443c-8487-509f22d9fc01)

```python
import numpy as np
import matplotlib.pyplot as plt

# Population proportion
population_proportion = 0.6

# Sample size
sample_size = 100

# Number of samples
num_samples = 1000

# Generate the population (for simulation purposes)
np.random.seed(0)

# Draw multiple samples and calculate the sample proportions
sample_proportions = []
for _ in range(num_samples):
    sample = np.random.binomial(1, population_proportion, sample_size)
    sample_proportion = np.mean(sample)
    sample_proportions.append(sample_proportion)

# Plot the histogram of the sample proportions
plt.hist(sample_proportions, bins=30, edgecolor='black', alpha=0.7, density=True)
plt.axvline(population_proportion, color='red', linestyle='dashed', linewidth=1)
plt.title('Sampling Distribution of the Sample Proportion')
plt.xlabel('Sample Proportion')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# Print the mean and standard deviation of the sampling distribution
mean_of_sample_proportions = np.mean(sample_proportions)
std_dev_of_sample_proportions = np.std(sample_proportions)

print(f"Mean of the sampling distribution: {mean_of_sample_proportions:.4f}")
print(f"Standard deviation of the sampling distribution (Standard Error): {std_dev_of_sample_proportions:.4f}")
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e5b1093e-3b35-4b3a-bda0-fe7b25df4a6f)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/cfeb4128-6160-4bbd-af76-371f4a217a24)




