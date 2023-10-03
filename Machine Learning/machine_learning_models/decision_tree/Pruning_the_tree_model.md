The Procedure of Pruning a Tree Model:

Select the candidate alphas:
1. By using the whole dataset, include training, validation and testing dataset, to train a full tree model with alpha = 0
2. Increase the alpha value until pruning the full tree model can result in lower tree score
3. Continue increase the alpha until only one leaf left
4. These alpha will be the candidate alpha, and we will use cross validation to find the optimal alpha

Hyperparameter Tuning for Alpha with Cross Validation:
1. By using the training dataset, create a full tree model
2. Pruning this full tree model to get a bunch of sub tree model
3. Select the optimal tree model that has the lowest tree score with different candidate alpha
4. Using the validate dataset to calculate SSR for each optimal tree model with different candidate alpha
5. Select the alpha with lowest SSR
6. Keep above process for each fold, and select the alpha with most vote

Model building and testing:
1. Combine traning dataset and validation dataset, and build a full tree model, and prune it into a bunch of sub tree model
2. Using the optimal alpha found in cross validation process to select the best tree model with lowest tree score
3. Using the test dataset to check the model performance

For Details: https://github.com/yangshiteng/Data-Science-Learning-Path/blob/main/Machine%20Learning/machine_learning_models/decision_tree/prunning.pdf
