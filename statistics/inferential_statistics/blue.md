![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b14dce05-5f95-433b-8f07-260547dc0aa2)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/5efaff25-bfd3-4c71-bfc5-42cb1b415586)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0311b9f4-2f81-4318-9317-b3a091fe1182)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a8e0d3c1-56b4-462e-a3a5-aa6d1a7e76b7)

```python
import numpy as np
import statsmodels.api as sm

# Data: Hours studied and exam scores
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
exam_scores = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

# Add a constant to the independent variable (for the intercept)
X = sm.add_constant(hours_studied)

# Perform OLS regression
model = sm.OLS(exam_scores, X)
results = model.fit()

# Print the results
print(results.summary())

# Extract the estimated coefficients
beta_0 = results.params[0]  # Intercept
beta_1 = results.params[1]  # Slope

print(f"Estimated Intercept (beta_0): {beta_0:.2f}")
print(f"Estimated Slope (beta_1): {beta_1:.2f}")
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/39676ece-61d0-45ec-a62e-84b875cc82ad)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/28cc63cd-4d9f-43d4-aa75-833a656fc314)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/870ca701-0c9e-4386-b4f1-3d6b35e52772)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/58489585-5677-494d-adc1-4aa6c56b00e7)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c7902150-4e76-498a-a4ea-e2c13fe547e4)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/96d5309c-9c36-4d4c-9419-41dfa244eec6)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/dd8d1c5f-77eb-4570-914c-52d5a4c1996a)

