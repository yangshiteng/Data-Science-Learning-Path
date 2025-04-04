# Confusion Matrix

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/046cba68-72a5-4e78-9911-ef58a5d459d0)

# Sensitivity (also known as Recall or True Positive Rate)

* The proportion of the actual positive cases that can be correctly classified as positive by the model
* It is a metric used to measure how well the model is in correctly identifying the actual positive cases
* For example, if the sensitivity is 85%, this means that, among all the actual positive cases, 85% of them can be correctly identified as positive by the model
* In another words, if a case is actual positive, there is 85% probability such that this case will be classified as positive by the model

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3bd58fb7-1164-488d-bd8a-115561f801bf)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b65466f2-fbc6-4b00-953d-a3efbadefc0c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/43d38e3c-f1dd-46fb-afe3-5aa60f9a5e97)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/358fa37b-1196-4eaa-8a10-bd90130d3dac)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2166f913-0ad4-4519-a137-4e8887bc610b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f5debf69-df39-44e4-abdd-f984a0ca759d)

# Specificity (True Negative Rate)

* The proportion of the actual negative cases that can be correctly classified as negative by the model
* It is a metric used to measure how well the model is in correctly identifying the actual negative cases
* For example, if the specificity is 96%, this means that, among all the actual negative cases, 96% of them can be correctly identified as negative by the model
* In another words, if a case is actual negative, there is 96% probability such that this case will be classified as negative by the model

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/57421d6a-74bb-430c-99c5-0fc73672edcf)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2340c0eb-d990-478d-a53b-796ee76e092b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0bac6eb9-2c25-45e1-806d-0463a87a2036)

# False Positive Rate (1 - Specificity)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a18a7661-fef8-47a5-a283-1f909abd8108)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d5047817-e13c-4e6a-a332-f66edbcc824d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/22a3fc81-502d-463d-af97-ea7beb345c55)

# Precision

* The proportion of the predicted positive cases that are actual positive
* It is a metric used to measure how accurate and reliable the model is in making positive prediction
* For example, if the precision is 87%, this means that, among all the predicted positive cases, 87% of them are actual positive
* In another words, if a case is predicted as positive, there is 87% probability such that this predcition is correct

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f06e806c-d63c-41e2-8fa2-775028e29a57)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/03f934a6-4e4c-4068-8fd7-f1b0312b994d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/56542d9a-69d1-4201-b08a-fa8ca1dbb63e)

# Negative Predictive Value (NPV)

* The proportion of the predicted negative cases that are actual negative
* It is a metric used to measure how accurate and reliable the model is in making negative prediction
* For example, if the NPV is 86%, this means that, among all the predictived negative cases, 86% of them are actual negative
* In another words, if a case is predicted as negative, there is 86% probability such that this prediction is correct

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/065d1be3-5b12-4621-a052-b75c768694a5)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6a1ce50e-2ccb-4cea-b1ff-27881a520cf1)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9ff3585f-0dd6-47cb-ab9e-5b3d365f3a02)

# Receiver Operating Characteristic (ROC) Curve (for balanced dataset)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ddc3a801-8d5a-4c2c-be88-086f35561854)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4a1d8c68-caab-4813-96f3-bb8a0a62f166)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3f54e158-4f68-4d8c-a7fc-5e16a342086b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/8efd62df-f59a-40af-b1db-025f8ec32b34)

# Precision-Recall (PR) Curve (for imbalanced dataset)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2add4d21-1e32-4512-a344-18e7f24aae78)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/76b980c9-e1e9-479e-968f-f7b37627d3de)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/796a1b7f-33f7-4a92-851d-f4ebd73c3508)

# F1 Score 

* Combine the precision and the recall into a single metric by the way of Harmonic mean
* Model comparision for imbalanced dataset considering both false positive and false negative

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/bee9bfa2-20fd-4e09-ad1e-30abf95a8a75)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b26bf2c4-22be-463f-b116-cae996d03e73)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f5eb1453-9dc4-4744-b9d7-9497bde48455)














