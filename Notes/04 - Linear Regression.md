![image](https://user-images.githubusercontent.com/60442877/187084761-cc69bdf3-0028-4249-bbda-789dcd9e7dc4.png)

![image](https://user-images.githubusercontent.com/60442877/193370553-48338ff4-c108-4f5b-b3ce-8d88f5b1e5cd.png)

![image](https://user-images.githubusercontent.com/60442877/193370614-8cbfb22e-5df8-40fa-807b-161175f28db9.png)

![image](https://user-images.githubusercontent.com/60442877/193370771-ac744a32-7927-41f8-9c51-8c93fe4f57a2.png)

[Overall Significance Test of Regression Model](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/Notes/F-test%20(Overall%20Significance%20Test%20of%20Regression%20Model).md)

![image](https://user-images.githubusercontent.com/60442877/187084817-eff50189-5165-42d7-97bc-cee75fd522d0.png)

![image](https://user-images.githubusercontent.com/60442877/187084965-decb9f54-a7e2-4834-90be-d5bf29c6a730.png)

![image](https://user-images.githubusercontent.com/60442877/187086275-15a668d4-ccf4-4458-ae2a-1a0c8535d6e5.png)

![image](https://user-images.githubusercontent.com/60442877/187086911-0645e060-1446-4142-99d4-ca469fb1e13d.png)

![image](https://user-images.githubusercontent.com/60442877/187087115-f382b5a5-06be-48ee-b7ac-99f09bf3b161.png)

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


# Analytic Solution of Linear Regression

![image](https://user-images.githubusercontent.com/60442877/188330863-962fb627-ae80-46c7-bf08-764d5a2bce91.png)

![image](https://user-images.githubusercontent.com/60442877/188330858-ba32f176-0f3b-4b3c-9739-fb71539c7b75.png)






