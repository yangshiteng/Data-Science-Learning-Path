# Measures of Variability: Range, Variance, Standard Deviation, Interquartile Range

Measures of variability describe the spread of data points within a dataset. These measures help to understand the degree of variation or dispersion from the average (mean) or from one another. Here are key measures of variability:

## 1. Range
- **Definition**: The range is the difference between the maximum and minimum values in a dataset.
- **Calculation**: 
  - `Range = Maximum value - Minimum value`
- **Example**: 
  - Data: [4, 7, 15, 21, 34]
  - Range: 34 - 4 = 30

## 2. Variance
- **Definition**: Variance measures the average squared deviations from the mean. It quantifies how far each number in the set is from the mean and thus from every other number in the set.
- **Calculation**: 
  - `Variance (σ²) = (Σ(xᵢ - μ)²) / n`
  - Where `xᵢ` is each value, `μ` is the mean, and `n` is the number of values.
- **Example**:
  - Data: [5, 7, 9, 11]
  - Mean: 8
  - Variance: ((5-8)² + (7-8)² + (9-8)² + (11-8)²) / 4 = 5

## 3. Standard Deviation
- **Definition**: Standard deviation is the square root of the variance and provides a measure of the spread of data points around the mean in the units of the data.
- **Calculation**:
  - `Standard Deviation (σ) = √Variance`
- **Example**:
  - Using the variance example above, the standard deviation would be √5 ≈ 2.24.

## 4. Interquartile Range (IQR)
- **Definition**: The interquartile range is the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the data. It measures the spread of the middle 50% of the data.
- **Calculation**:
  - `IQR = Q3 - Q1`
- **Example**:
  - Data: [3, 5, 7, 8, 9, 11, 14, 16, 18]
  - Q1 (25th percentile): 7
  - Q3 (75th percentile): 14
  - IQR: 14 - 7 = 7

## Conclusion

These measures of variability are essential for providing a complete picture of the data. They are particularly useful in research, analytics, and other statistical analyses to understand variability, which is as important as understanding the average.
