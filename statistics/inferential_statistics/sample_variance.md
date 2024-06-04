![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2d1b7904-c81d-4e30-a56f-1495063cdd2b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e415152f-294d-4629-b02c-e232e37f3f77)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ae1e5b92-608a-46b0-8659-6fc26f5ff603)

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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

# Draw multiple samples and calculate the sample variances
sample_variances = []
for _ in range(num_samples):
    sample = np.random.choice(population, sample_size)
    sample_variances.append(np.var(sample, ddof=1))

# Plot the histogram of the sample variances
plt.hist(sample_variances, bins=30, edgecolor='black', alpha=0.7, density=True)
plt.title('Sampling Distribution of the Sample Variance')
plt.xlabel('Sample Variance')
plt.ylabel('Frequency')

# Plot the theoretical chi-square distribution
df = sample_size - 1
chi2_values = np.linspace(min(sample_variances), max(sample_variances), 100)
chi2_pdf = stats.chi2.pdf(chi2_values * (sample_size - 1) / population_std_dev**2, df) * (sample_size - 1) / population_std_dev**2
plt.plot(chi2_values, chi2_pdf, 'r-', lw=2, label='Theoretical Chi-square Distribution')
plt.legend()

# Show the plot
plt.show()

# Print the mean and variance of the sampling distribution
mean_of_sample_variances = np.mean(sample_variances)
std_dev_of_sample_variances = np.std(sample_variances)

print(f"Mean of the sampling distribution: {mean_of_sample_variances:.2f}")
print(f"Standard deviation of the sampling distribution: {std_dev_of_sample_variances:.2f}")
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/435ac32c-fa01-4061-9cb0-be9df03614c9)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6a84e60b-8f79-4b57-b0d0-abeca60c7cbe)







