# Gradient Boost for Regression
### 1. Regression Gradient Boost starts by making a tree model with only one leaf, this leaf represents an initial prediction for the continuous dependent variable and is just the average value
### 2. Build a restricted tree model (for example, only allow up to 4 leaves) based on the errors made by the previous tree model, specifically, build a restricted tree model based on the residuals calculated from the difference between the observed value of dependent variable and the predicted value from previous tree model
### 3. The output of new tree model is just the average value of each leaf
### 4. The new predicted value is calculated by, the initial prediciton + learning rate * output of first tree
### 5. Keep making trees based on the errors made by the previous tree until we reach the maximum specified or adding additional trees does not significantly reduce the size of the residuals

# Gradient Boost for Classification (binary classification case)
### 1. Unlike the Gradient Boost for regression which can directly predicts the continuous value for dependent variable, the Gradient Boost for classification predicts the log(odds) first, then this log(odds) can be converted into the probability by using logistic function, and we use this probability to make the classification
### 2. Classification Gradient Boost starts by making a tree model with only one leaf, this leaf represents an initial prediction for the log(odds) which can be converted into the probability of being positive by using logistic function. The initial prediciton for the log(odds) is just the logarithm of the ratio of the number of positive over the number of negative labels
![image](https://user-images.githubusercontent.com/60442877/195940400-0f155491-bc43-45f6-ac87-c8a33f7dc117.png)
### 3. By using the logistic function, we can convert this initial prediction of log(odds) into the probability of being positive. Then, we calculate the "Pseudo Residual" and build the tree model with some restriction (only allow up to 4 leaves for example). The predicted value is just the probability converted from logistic function and the observed value is either 1 for positive label or 0 for negative label
![image](https://user-images.githubusercontent.com/60442877/195945538-83199091-c9a1-47fe-9649-8d8820651789.png)
![image](https://user-images.githubusercontent.com/60442877/195945578-24f04fe4-64e7-46bf-84fa-ed10667a4087.png)
### 4. The output in each leaf of the tree model can be obtain by the following formula:
![image](https://user-images.githubusercontent.com/60442877/195947664-51a6e8d4-73d3-402b-bd21-30e6bc1e9371.png)
### 5. The new prediction of log(odds) can be obtain by initial prediction of log(odds) + learning rate * the output of first tree model
![image](https://user-images.githubusercontent.com/60442877/195948065-de107de2-6303-40a6-b381-1efbf20a1406.png)
### 6. Repeat above process until we have made the maximum number of trees specified, or the residuals get super small

