# XGBoost for Regression Summary

1. The very first step in fitting XGBoost to the training dataset is to make an initial prediction. This prediction can be anything, but by default it is 0.5 regardless of whether you are using XGBoost for Regression or Classification
2. Calculate the residuals for each sample in training dataset which is just the difference between the observed value and the predicted value (in this case, the initial prediction 0.5)
3. Fitting the XGBoost Tree to the residuals, and instead of using Gini index to select threshold, this time, we use something called "Similarity Score" and "Gain" to help us select the threshold
![image](https://user-images.githubusercontent.com/60442877/196044677-fa221198-b97b-471e-9467-76a3851cb117.png)
