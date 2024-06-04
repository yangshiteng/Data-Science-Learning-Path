![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/45cad2c4-9fae-4e82-ac43-6b603b552193)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ae266db3-615b-4fa9-bfad-e3dfb2080aca)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c92c667c-655b-4da5-b73d-88cceb7d6ad8)

```python
import numpy as np
import matplotlib.pyplot as plt

# Population parameters
population_mean = 75
population_std_dev = 10

# Sample size
sample_size = 30

# Number of samples
num_samples = 1000

# Generate the population (for simulation purposes)
np.random.seed(0)
population = np.random.normal(population_mean, population_std_dev, 100000)

# Draw multiple samples and calculate the sample means
sample_means = []
for _ in range(num_samples):
    sample = np.random.choice(population, sample_size)
    sample_means.append(np.mean(sample))

# Plot the histogram of the sample means
plt.hist(sample_means, bins=30, edgecolor='black', alpha=0.7, density=True)
plt.axvline(population_mean, color='red', linestyle='dashed', linewidth=1)
plt.title('Sampling Distribution of the Sample Mean')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# Print the mean and standard deviation of the sampling distribution
mean_of_sample_means = np.mean(sample_means)
std_dev_of_sample_means = np.std(sample_means)

print(f"Mean of the sampling distribution: {mean_of_sample_means:.2f}")
print(f"Standard deviation of the sampling distribution (Standard Error): {std_dev_of_sample_means:.2f}")
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a2170ce8-8f76-4974-bf49-4f3a8850fa3c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a46e4dd5-dca3-40d2-968c-f296a82151ad)













