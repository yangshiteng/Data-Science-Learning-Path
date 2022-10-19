# XGBoost for Regression Summary (directly predict continuous value for dependent variable)

1. The very first step is to make an initial prediction. This prediction can be anything, but by default it is 0.5 regardless of whether you are using XGBoost for Regression or Classification
2. Calculate the residuals for each sample in training dataset which is just the difference between the observed value and the predicted value (in this case, the initial prediction 0.5)
3. Fitting the XGBoost Tree model to these residuals, and instead of using Gini index to select threshold, this time, we use something called "Similarity Score" and "Gain" to help us select the threshold.
![image](https://user-images.githubusercontent.com/60442877/196044677-fa221198-b97b-471e-9467-76a3851cb117.png)
![image](https://user-images.githubusercontent.com/60442877/196045937-5633a8e1-46cb-4a67-a677-a7f287c02a53.png)
4. Pruning the tree by calculating the difference between gain and pruning parameter (gamma), if positive, keep branch, if negative, remove it
5. Calculate the output of XGBoost tree for regression
![image](https://user-images.githubusercontent.com/60442877/196046257-83a4ee0d-f54f-4c4a-84b0-27f1c23fe03f.png)
6. Calculate the new predicted value which is equal to the initial prediction + learning rate * the output of first tree
7. Repeat above process to keep building trees until the residuals are super small or we have reached the maximum number

# XGBoost for Classification Summary (predict log(odds) which can be converted to probability by logistic function)

1. The very first step is to make an initial prediction. This prediction can be anything, for example, the probability of observing positive labels in the training dataset, but by default it is 0.5 regardless of whether you are using XGBoost for Regression or Classification. Since this is for classification, you also need to convert this initial probability prediction to log(odds)
2. Calculate the residuals for each sample in training dataset which is just the difference between the observed value (1 if positive, 0 if negative) and the predicted value (the predicted probability converted from log(odds))
3. Fitting the XGBoost Tree model to these residuals, and instead of using Gini index to select threshold, this time, we use something called "Similarity Score" and "Gain" to help us select the threshold
![image](https://user-images.githubusercontent.com/60442877/196326643-3e3b4b82-af68-4448-8b5c-7e7390ade7b6.png)
4. Pruning the tree by calculating the difference between gain and pruning parameter (gamma), if positive, keep branch, if negative, remove it
5. Calculate the output of XGBoost tree for classification
![image](https://user-images.githubusercontent.com/60442877/196327045-8cb89f38-0123-403f-8cc4-388d890eab6d.png)
6. Calculate the new predicted log(odds) value which is equal to the initial prediction of log(odds) + learning rate * the output of first tree
7. Repeat above process to keep building trees until the residuals are super small or we have reached the maximum number

# Future Learning Plan

https://machinelearningmastery.com/xgboost-with-python/

