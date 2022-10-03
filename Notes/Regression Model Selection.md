# Introduction

Model selection is the process of selecting a model from a set of candidate models. Many statistical techniques involve model selection either implicitly or explicitly: e.g., hypothesis tests require selecting between a null hypothesis and alternative hypothesis model; an autoregressive model requires selecting the order p; in this tutorial, a regression model requires selecting specific predictors.

A good model selection technique will balance goodness of fit with simplicity, which means under a certain performance requirement set by researchers, the best model should be as simple as possible. If a model includes too many predictors, the most common issue would be overfitting, in which case the model gives good predictions to training data but performs much worse when it comes to data not used in fitting the model. Overfitting model normally gives low bias but high variance. Conversely, when there are too few predictors in a model, the issue of underfitting may show up, and gives poor predictions to data used or not used for model fitting. Underfitting models normally have low variance but high bias.

![image](https://user-images.githubusercontent.com/60442877/188525133-522b289f-b27d-4b65-b862-be6a21e9f49e.png)

## Testing based and criterion-based approaches are the two main approaches for model (variable) selection

Testing-based approaches include backward elimination, forward selection, stepwise regression, etc. In this category, variables are selected based on whether they are significant or not when they are added/removed. 

For criterion-based approaches, we have some idea about the purpose for which a model is intended, so we might propose some measures of how well a given model meets that purpose. Then a model that optimizes a criterion which balances goodness-of-fit will be chosen. Some examples of criterion-based approaches include AIC/BIC, adjusted R2, Mallow’s cp, etc.

![image](https://user-images.githubusercontent.com/60442877/188523263-5a3b92da-6a47-46af-b89d-438a715018fa.png)

![image](https://user-images.githubusercontent.com/60442877/188523271-0f879d59-3fed-4e44-bf1d-65a70eb3003d.png)

## F-test

![image](https://user-images.githubusercontent.com/60442877/188527552-ed08f9d0-ea71-48b7-b948-6de3cb3b92f4.png)

## Likelihood Ratio Test (LRT)

![image](https://user-images.githubusercontent.com/60442877/188527596-07c02240-ef71-48a4-a7bd-cf57b13da48c.png)

## Compare Models with same size

* RSS (Residual Sum of Squared)
* R-squared

## Compare Models with different size

### AIC and BIC

![image](https://user-images.githubusercontent.com/60442877/188527617-b468ef97-06d1-4730-be7a-ed1f8bccebd8.png)

### Adjusted R-squared

![image](https://user-images.githubusercontent.com/60442877/188528451-cd57d854-2c96-440e-9c47-9586849aadf9.png)

* Adjusted R-squared can help you avoid the fundamental problem with regular R-squared which always increases when you add an independent variable
* Adjusted R-squared increases only when a new variable improves the model by more than chance. Low-quality variables can cause it to decrease.

## Model Selection Method

### Stepwise Selection

![image](https://user-images.githubusercontent.com/60442877/188527678-b5968a0d-c962-4301-a9ef-69fe16fa223d.png)

### Best Subsets Selection

![image](https://user-images.githubusercontent.com/60442877/188528876-35e04993-71af-471d-9513-8d2de07a10a8.png)

![image](https://user-images.githubusercontent.com/60442877/188528904-b7fa1f69-b075-43bb-8f40-2d4927e25cbb.png)

![image](https://user-images.githubusercontent.com/60442877/188529130-c99e18f5-d9c3-4292-9ae2-05ffb1823851.png)

![image](https://user-images.githubusercontent.com/60442877/188529192-38bf1807-bf45-4388-8161-b235095abe9c.png)

![image](https://user-images.githubusercontent.com/60442877/188529252-293080cd-dda7-4dff-9802-9b40db63bf50.png)

![image](https://user-images.githubusercontent.com/60442877/188529356-bfcbcf3b-0231-4f88-8b0b-81494de5b283.png)

![image](https://user-images.githubusercontent.com/60442877/188529388-9ca9a253-a8fe-4e4f-a42c-fe228b7d054d.png)

![image](https://user-images.githubusercontent.com/60442877/188529404-9d4f36a7-b198-48e5-94e1-19a42f062b0d.png)

![image](https://user-images.githubusercontent.com/60442877/188529522-74b499ab-3138-435a-bba9-bb1c1e2f8caa.png)


