# AdaBoost for Classfication

1. Give each sample a weight that indicates how important it is to be correctly classified, the initial weights are the same for all the samples, be just 1 over the total number of samples
2. By using Gini index, build the stump
3. Calculate the total error, which is the sum of the weights associated with the incorrectly classified samples
4. Calculate the amount of say, which is equal to one half of the logarithm of the ratio of (1 - total error) over total error
![image](https://user-images.githubusercontent.com/60442877/194757433-fc11272e-6e97-4c38-a744-ce60cfc527a4.png)
5. Calculate the new weights for samples that are incorreclty classified
![image](https://user-images.githubusercontent.com/60442877/194757665-244e2613-d672-43e7-99c7-7705f9fa91a3.png)
6. Calculate the new weights for samples that are correctly classified
![image](https://user-images.githubusercontent.com/60442877/194757694-7480a0f3-94de-4644-a05f-694dc679287c.png)
7. Normalize these new sample weights so that they add up to 1
8. With these new sample weights, by using weighted gini index, create the second stump, then, repeat 3-7 procedure (Or instead of weighted gini index, we can make a new dataset of same size with bootstrap sampling by using those new sample weights, and give each sample in this new dataset same weight, and repeat 1-7 procedure)

![image](https://user-images.githubusercontent.com/60442877/194981873-a1d1c1e9-a536-4f36-994e-d93f3eb254e7.png)

# AdaBoost for Regression

![image](https://user-images.githubusercontent.com/60442877/235336026-d7696596-9486-4af6-8131-616741aa4c19.png)
