![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3f6fd150-1250-4dfc-a708-08fae2591be6)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/609c0da7-b405-4de8-880c-5e54fd5be84b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/31a6b6b8-00ec-4e93-a314-6e33861ffdcd)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/51140271-e14c-4c27-97c1-ad8ddd1c6862)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/392ac473-1ef0-4579-aae1-ca0ab9c83bb5)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/04047ecb-7c48-4cca-b983-f095e1d8cb0b)

```python
import numpy as np

# Data: Exam Scores
data = np.array([85, 90, 78, 92, 88, 76, 95, 89, 84, 91])

# Sample mean
sample_mean = np.mean(data)

# Sample standard deviation (ddof=1 for sample standard deviation)
sample_std_dev = np.std(data, ddof=1)

# Sample size
n = len(data)

# Standard error of the mean
standard_error = sample_std_dev / np.sqrt(n)

print(f"Sample Mean: {sample_mean:.2f}")
print(f"Sample Standard Deviation: {sample_std_dev:.2f}")
print(f"Standard Error of the Mean: {standard_error:.2f}")
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a1e8b64b-62ad-4e69-b0a2-f8a79b54bedd)
