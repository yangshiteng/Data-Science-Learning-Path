# What is a Scatter Plot?
A scatter plot is a graphical representation that uses dots to represent the values obtained for two different variables, one plotted along the x-axis and the other along the y-axis. Each point on the plot corresponds to one observation in the dataset.

# Key Characteristics of Scatter Plots:
- **Points**: Each point in the scatter plot represents an observation in the dataset with its position determined by the values of the two variables.
- **Axes**: The x-axis typically represents the independent variable, while the y-axis represents the dependent variable.
- **Trends**: The pattern of the points in the plot can indicate the type of relationship between the two variables, such as linear, nonlinear, or no correlation.

# Uses of Scatter Plots:
- To identify the type of relationship between two variables (e.g., positive, negative, or no correlation).
- To detect outliers or unusual data points in the dataset.
- To assess the strength and direction of the relationship between two continuous variables.

# Examples with Python Code

## Basic Scatter Plot Example
This example creates a simple scatter plot using Python's Matplotlib library to show the relationship between two sets of data.

```python
import matplotlib.pyplot as plt

# Sample data
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 100, 86, 103, 87, 94, 78, 77, 85, 86]

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.title('Basic Scatter Plot')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/565a45b4-74d1-42cc-a132-dc7c8cb2fd22)

## Scatter Plot with Trend Line
To further analyze the relationship, you can add a trend line. Here’s how you add a linear trend line to a scatter plot:

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6])
y = np.array([99, 86, 87, 88, 100, 86, 103, 87, 94, 78, 77, 85, 86])

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue')

# Calculating and adding the trend line
m, b = np.polyfit(x, y, 1)  # m = slope, b = intercept
plt.plot(x, m*x + b, color='red')

plt.title('Scatter Plot with Trend Line')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d6fe6584-b026-4280-b6ec-5a4fac4240b1)

## Comparative Scatter Plots
For comparative purposes, here’s how to plot multiple scatter plots in the same figure, which is useful for comparing different datasets.

```python
import matplotlib.pyplot as plt

# Data for two groups
x1 = [5, 7, 8, 5, 2, 15, 2, 9, 4, 11, 12, 9, 6]
y1 = [99, 86, 87, 88, 77, 86, 85, 87, 90, 78, 77, 85, 86]
x2 = [2, 3, 4, 3, 6, 11, 12, 9, 6, 1, 0, 3, 4]
y2 = [100, 105, 84, 105, 90, 99, 100, 88, 95, 95, 80, 85, 92]

# Plotting two scatter plots in the same figure
plt.figure(figsize=(10, 6))
plt.scatter(x1, y1, color='blue', label='Group 1')
plt.scatter(x2, y2, color='green', label='Group 2')
plt.title('Comparative Scatter Plots')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')
plt.legend()
plt
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/225010e9-a7b8-4d12-a6d8-053cc23fc5b2)




