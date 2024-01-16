# Introduction

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/340f7649-081f-419b-89bb-c093f97da59c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e93367b3-c5de-4dfc-a8f2-cc8d0cd59b25)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2ed40f96-168d-4a11-9ea5-33e189714f1c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9b88f4b7-115c-43d3-93c0-422169845aa2)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f04f6874-d6ab-4244-9d1b-82f34be8440d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9ce0edf3-6f18-4134-8758-5a4ab28bd57f)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e0b10916-c01d-42ac-aa17-fc56b94c829c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3802f81c-31ad-490f-a02a-323f764cfb91)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/76767e96-cb78-4d92-a11b-76c266c40993)

# Matrix Formulation

![image](https://user-images.githubusercontent.com/60442877/147891201-066a731d-6e34-4fdc-abce-7ab0e6420572.png)

![image](https://user-images.githubusercontent.com/60442877/188330858-ba32f176-0f3b-4b3c-9739-fb71539c7b75.png)

# Linear Regression 5 Assumptions

## 1. Linearity

The relationship between the independent and dependent variables must be linear.  It is also important to check for outliers since linear regression is sensitive to outlier effects.  The linearity assumption can best be tested with scatter plots, and it can be measured by pearson correlation coefficient 

Note: if pearson correlation coefficient is 0, it only means that there is no linear relationship between two variables, but these two variables may have some non-linear relationship.

## 2. Homoscedasticity (同方差性)

The error term in each observation should have same variance. The scatter plot is good way to check whether the data are homoscedasticity or heteroscedasticity. The following scatter plots show examples of data that are not homoscedasticity. 

![image](https://user-images.githubusercontent.com/60442877/188330242-080c8718-1dd5-4d0c-a241-7bce3c8977fa.png)

The Goldfeld-Quandt Test can also be used to test for heteroscedasticity.  The test splits the data into two groups and tests to see if the variances of the residuals are similar across the groups.

## 3. Independence

Observations are independent of each other.
 
## 4. Normality

The linear regression analysis requires all variables to be multivariate normal distributed. This assumption can best be checked with a histogram or a Q-Q-Plot.  Normality can be checked with a goodness of fit test, e.g., the Kolmogorov-Smirnov (KS) test.  When the data is not normally distributed, a non-linear transformation (e.g., log-transformation) might fix this issue.

## 5. No Multicollinearity

Linear regression assumes that there is little or no multicollinearity in the data.  Multicollinearity occurs when the independent variables are too highly correlated with each other. If the degree of correlation between variables is high enough, it can cause problems when you fit the model and interpret the results.

A key goal of regression analysis is to isolate the relationship between each independent variable and the dependent variable. The interpretation of a regression coefficient is that it represents the mean change in the dependent variable for each 1 unit change in an independent variable when you hold all of the other independent variables constant. The idea is that you can change the value of one independent variable and not the others. However, when independent variables are correlated, it indicates that changes in one variable are associated with shifts in another variable. The stronger the correlation, the more difficult it is to change one variable without changing another. It becomes difficult for the model to estimate the relationship between each independent variable and the dependent variable independently because the independent variables tend to change in unison.

Moreover, suppose two independent variables are higly correlated, for example, X1 = 2 * X2, then, if we take a look at matrix X, it is no longer a full rank matrix and this will make the analytical solution unavailable. 

Multicollinearity may be tested with three central criteria:

(1) Correlation matrix – when computing the matrix of Pearson Correlation Coefficient among all independent variables, the correlation coefficients need to be smaller than 1

(2) Tolerance – the tolerance measures the influence of one independent variable on all other independent variables. With T < 0.1 there might be multicollinearity in the data and with T < 0.01 there certainly is.

![image](https://user-images.githubusercontent.com/60442877/188330631-0a33a3a8-665e-4e86-8aa6-855fa069bb3a.png)

(3) Variance Inflation Factor (VIF) – the variance inflation factor of the linear regression is defined as VIF = 1/T. With VIF > 5 there is an indication that multicollinearity may be present; with VIF > 10 there is certainly multicollinearity among the variables












