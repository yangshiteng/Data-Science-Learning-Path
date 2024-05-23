![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2020d14d-2349-4054-a767-3bf4672c9a73)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b112c7a1-eded-4f3b-80a2-a2d10cb23964)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Parameters
k = 2  # shape parameter
theta = 3  # scale parameter

# Generate random samples
data = gamma.rvs(a=k, scale=theta, size=1000)

# Plot the histogram of the data
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

# Plot the Gamma distribution PDF
x = np.linspace(0, np.max(data), 1000)
pdf = gamma.pdf(x, a=k, scale=theta)
plt.plot(x, pdf, 'r-', lw=2, label='Gamma PDF')

# Labels and title
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Gamma Distribution (k=2, θ=3)')
plt.legend()
plt.grid(True)
plt.show()

```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/cf52f080-10b9-43c5-9b5d-3a353b916f67)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b4aa783d-7fd2-4709-afa8-533f8e05fa3f)
