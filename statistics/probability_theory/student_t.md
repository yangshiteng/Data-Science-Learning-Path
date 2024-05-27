![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a9992bb8-5da9-4654-a8e5-b4c37354113d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/83697319-d90b-490b-9686-696dbc091d53)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/48b1e247-f6d3-4d43-a77a-9e4d2b33e2b6)

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
df = 9  # degrees of freedom

# Create a t-distribution
t_dist = stats.t(df)

# Generate random samples
samples = t_dist.rvs(size=1000)

# Calculate mean and variance
mean = np.mean(samples)
variance = np.var(samples)

print(f"Sample Mean: {mean}")
print(f"Sample Variance: {variance}")

# Plot the PDF
x = np.linspace(-4, 4, 1000)
pdf = t_dist.pdf(x)
plt.plot(x, pdf, label='PDF')

# Plot the histogram of the samples
plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Histogram of samples')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Student\'s t-Distribution PDF')
plt.grid(True)
plt.show()

# Plot the CDF
cdf = t_dist.cdf(x)
plt.figure()
plt.plot(x, cdf, label='CDF')
plt.legend()
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.title('Student\'s t-Distribution CDF')
plt.grid(True)
plt.show()

```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/08adb5f6-30bb-448d-9492-9843b4729dcc)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/482f6561-ebfc-4abc-9772-da4dce3099b7)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d01a5915-cd1a-4256-a12f-d98f46aca6aa)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/fb689853-462d-493a-80e1-1b6f12006e2c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d4f84c38-bcfd-461c-ab2c-469eb12a04b3)

```python
import numpy as np
import scipy.stats as stats

# Sample statistics
sample_mean = 10
sample_std = 2
n = 15

# Degrees of freedom
df = n - 1

# Confidence level
confidence_level = 0.95

# Calculate the critical value
alpha = 1 - confidence_level
t_critical = stats.t.ppf(1 - alpha/2, df)

# Calculate the margin of error
margin_of_error = t_critical * (sample_std / np.sqrt(n))

# Calculate the confidence interval
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print(f"Sample Mean: {sample_mean}")
print(f"Sample Standard Deviation: {sample_std}")
print(f"Sample Size: {n}")
print(f"Degrees of Freedom: {df}")
print(f"Critical Value (t): {t_critical}")
print(f"Margin of Error: {margin_of_error}")
print(f"95% Confidence Interval: {confidence_interval}")

```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b903c15a-642e-484d-aac4-e726627457c6)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6a52f13f-c68b-4d23-ab14-85c3de7f1c60)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/02e374a2-4e13-4bec-a6a3-27f3d22fb2d1)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ffda85c0-e489-4a5e-a7d4-1e11a67c828c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/221f844e-f0b0-424d-bf38-2f1da0e48d7f)

```python
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

# Sample data: Hours studied (X) and exam scores (Y)
X = np.array([2, 3, 5, 7, 8, 10, 12, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36])
Y = np.array([52, 54, 57, 60, 62, 65, 67, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94])

# Add a constant to the independent variable
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(Y, X).fit()

# Get the estimated coefficients and standard errors
coefficients = model.params
standard_errors = model.bse

# Print the results
print(f"Intercept: {coefficients[0]}")
print(f"Slope: {coefficients[1]}")
print(f"Standard Error of the Slope: {standard_errors[1]}")

# Perform the hypothesis test
t_value = coefficients[1] / standard_errors[1]
degrees_of_freedom = len(X) - 2
critical_value = stats.t.ppf(1 - 0.025, degrees_of_freedom)

print(f"t-Statistic: {t_value}")
print(f"Degrees of Freedom: {degrees_of_freedom}")
print(f"Critical Value (two-tailed, 0.05 significance level): {critical_value}")

# Decision
if abs(t_value) > critical_value:
    print("Reject the null hypothesis. There is a significant relationship between hours studied and exam scores.")
else:
    print("Fail to reject the null hypothesis. There is no significant relationship between hours studied and exam scores.")

```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ff577a32-6b08-492c-b241-86e32676fbae)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/1fbce841-34cc-47d6-bb2b-4f56c76df951)











