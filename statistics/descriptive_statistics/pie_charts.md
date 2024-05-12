# What is a Pie Chart?
A pie chart is a circular statistical graphic divided into slices to illustrate numerical proportion. Each slice of the pie represents a category's contribution to the total sum. The size of each slice is proportional to the quantity it represents.

# Key Characteristics of Pie Charts:
- **Slices**: Each slice of the pie represents a different category. The area of each slice is proportional to the statistic it represents.
- **Proportions**: Pie charts are best used when you want to show relative sizes of parts to the whole. They are ideal for displaying percentage or proportional data.
- **Color Coding**: Different slices are often color-coded to help distinguish between categories visually.

# Uses of Pie Charts:
- To compare parts of a whole across different categories.
- Ideal for illustrating market shares, survey results, or any proportional distribution.
- Effective when the number of categories is small (generally 5 or fewer for clarity).

# Examples with Python Code

## Basic Pie Chart Example
This example uses Python's Matplotlib library to create a simple pie chart illustrating the distribution of a dataset.

```python
import matplotlib.pyplot as plt

# Data to plot
labels = ['Apple', 'Banana', 'Cherry', 'Date']
sizes = [15, 30, 45, 10]
colors = ['red', 'yellow', 'pink', 'brown']
explode = (0, 0, 0.1, 0)  # only "explode" the 3rd slice (i.e., 'Cherry')

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Pie Chart of Fruit Distribution')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c271b090-162a-4aa9-99b4-d884baa406c5)
