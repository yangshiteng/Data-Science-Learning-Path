![image](https://user-images.githubusercontent.com/60442877/193882146-659acf27-335c-4449-ad01-22760d4765ab.png)

![image](https://user-images.githubusercontent.com/60442877/193882213-5f2639d7-3ce9-4e24-961e-57910a377415.png)

![image](https://user-images.githubusercontent.com/60442877/193882356-df09a746-fa45-466e-acc3-3ed40b3bc451.png)

![image](https://user-images.githubusercontent.com/60442877/193882720-11b20d3c-9819-4bc0-82c3-9d98b6416cb3.png)

![image](https://user-images.githubusercontent.com/60442877/193882930-68a01f37-4f54-4ac5-959f-0274b4ccc259.png)

![image](https://user-images.githubusercontent.com/60442877/193883397-1a47f0f5-b328-4a91-b515-33b574472e1e.png)

![image](https://user-images.githubusercontent.com/60442877/193883459-1d133ad7-b869-4afa-80a1-a2a42981d28f.png)

![image](https://user-images.githubusercontent.com/60442877/193883615-dc910803-b5ee-488d-8aa8-dae6a4cfc94b.png)

*should do pseudocount

![image](https://user-images.githubusercontent.com/60442877/193884097-c4df7b14-2512-4a98-8be6-93fc409a7f9b.png)

![image](https://user-images.githubusercontent.com/60442877/193884188-b8d2b8a9-f7ba-430f-9dc5-2ca27ed71bca.png)

![image](https://user-images.githubusercontent.com/60442877/193884403-98962d68-3df7-4833-984c-bf6dea868e04.png)

![image](https://user-images.githubusercontent.com/60442877/193884499-e55aa349-25ef-4b8b-9550-6ff993c3ecbc.png)

![image](https://user-images.githubusercontent.com/60442877/193884726-a3017faf-4222-487a-b877-5919462af9c0.png)

    # load the iris dataset
    from sklearn.datasets import load_iris
    iris = load_iris()

    # store the feature matrix (X) and response vector (y)
    X = iris.data
    y = iris.target

    # splitting X and y into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    # training the model on training set
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # making predictions on the testing set
    y_pred = gnb.predict(X_test)

    # comparing actual response values (y_test) with predicted response values (y_pred)
    from sklearn import metrics
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
    
    # Gaussian Naive Bayes model accuracy(in %): 95.0
    
# Naive Bayes Application
1. Document classification 
2. Spam filtering
