![image](https://user-images.githubusercontent.com/60442877/187084761-cc69bdf3-0028-4249-bbda-789dcd9e7dc4.png)

![image](https://user-images.githubusercontent.com/60442877/187084817-eff50189-5165-42d7-97bc-cee75fd522d0.png)

![image](https://user-images.githubusercontent.com/60442877/187084965-decb9f54-a7e2-4834-90be-d5bf29c6a730.png)

![image](https://user-images.githubusercontent.com/60442877/187086275-15a668d4-ccf4-4458-ae2a-1a0c8535d6e5.png)

![image](https://user-images.githubusercontent.com/60442877/187086911-0645e060-1446-4142-99d4-ca469fb1e13d.png)

![image](https://user-images.githubusercontent.com/60442877/187087115-f382b5a5-06be-48ee-b7ac-99f09bf3b161.png)

# Linear Regression 5 Assumptions

1. Linearity

The relationship between the independent and dependent variables must be linear.  It is also important to check for outliers since linear regression is sensitive to outlier effects.  The linearity assumption can best be tested with scatter plots, and it can be measured by pearson correlation coefficient 

Note: if pearson correlation coefficient is 0, it only means that there is no linear relationship between two variables, but these two variables may have some non-linear relationship.

2. Homoscedasticity (同方差性)

The error term in each observation should have same variance. The scatter plot is good way to check whether the data are homoscedasticity or heteroscedasticity. The following scatter plots show examples of data that are not homoscedasticity. 

![image](https://user-images.githubusercontent.com/60442877/188330242-080c8718-1dd5-4d0c-a241-7bce3c8977fa.png)

The Goldfeld-Quandt Test can also be used to test for heteroscedasticity.  The test splits the data into two groups and tests to see if the variances of the residuals are similar across the groups.

3. Independence

Observations are independent of each other.
 
4. Normality

The linear regression analysis requires all variables to be multivariate normal distributed. This assumption can best be checked with a histogram or a Q-Q-Plot.  Normality can be checked with a goodness of fit test, e.g., the Kolmogorov-Smirnov (KS) test.  When the data is not normally distributed, a non-linear transformation (e.g., log-transformation) might fix this issue.

5. No Multicollinearity

Linear regression assumes that there is little or no multicollinearity in the data.  Multicollinearity occurs when the independent variables are too highly correlated with each other.

Multicollinearity may be tested with three central criteria:

(1) Correlation matrix – when computing the matrix of Pearson Correlation Coefficient among all independent variables, the correlation coefficients need to be smaller than 1

(2) Tolerance – the tolerance measures the influence of one independent variable on all other independent variables. With T < 0.1 there might be multicollinearity in the data and with T < 0.01 there certainly is.

(3) Variance Inflation Factor (VIF) – the variance inflation factor of the linear regression is defined as VIF = 1/T. With VIF > 5 there is an indication that multicollinearity may be present; with VIF > 10 there is certainly multicollinearity among the variables


# Analytic Solution of Simple Linear Regression
