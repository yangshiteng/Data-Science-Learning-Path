# Gradient Boost for Regression
1. Regression Gradient Boost starts by making a tree model with only one leaf, this leaf represents an initial prediction for the continuous dependent variable and is just the average value
2. Build a fixed size tree based on the errors made by the previous tree, specifically, build a tree model of fixed size based on the residuals calculated from the difference between the observed value of dependent variable and the predicted value from previous tree
3. The new predicted value is calculated by, the initial prediciton + learning rate * output of first tree
4. Keep making trees based on the errors made by the previous tree until we reach the maximum specified or adding additional trees does not significantly reduce the size of the residuals

