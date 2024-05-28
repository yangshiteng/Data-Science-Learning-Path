![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/00af1496-c819-4206-9b71-eb2116e1fca2)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/8111bea5-615f-4f83-94f2-15bc4d27a964)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/dab335f7-a188-406f-b0f1-f6d9681c0b73)

```python
import numpy as np
import matplotlib.pyplot as plt

# Given data
mean = 70
std_dev = 10
k = 2

# Chebyshev's bound
prob_outside = 1 / k**2
prob_inside = 1 - prob_outside

print(f"Probability that scores are within {k} standard deviations of the mean: {prob_inside:.2f}")

# Simulate a large number of exam scores with a normal distribution for visualization
np.random.seed(0)
exam_scores = np.random.normal(mean, std_dev, 10000)

# Calculate the proportion of scores within k standard deviations
within_k_std_dev = np.abs(exam_scores - mean) < k * std_dev
proportion_within_k_std_dev = np.mean(within_k_std_dev)

print(f"Proportion of simulated scores within {k} standard deviations of the mean: {proportion_within_k_std_dev:.2f}")

# Plot the histogram of the exam scores
plt.hist(exam_scores, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(mean, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean + k * std_dev, color='blue', linestyle='dashed', linewidth=1)
plt.axvline(mean - k * std_dev, color='blue', linestyle='dashed', linewidth=1)
plt.title('Histogram of Exam Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/1044dc6e-5151-4991-8aa6-74b98c02b29a)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/34048c97-e0da-482f-940d-ceb25a8cc9b9)
