![image](https://user-images.githubusercontent.com/60442877/235332187-2863f143-463d-4a01-9e7b-51e5ea35dec7.png)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import accuracy_score

    # Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create the AdaBoost classifier
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=42)

    # Train the model
    ada.fit(X_train, y_train)

    # Make predictions
    y_pred = ada.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

![image](https://user-images.githubusercontent.com/60442877/235332192-320b0d2a-ab6d-4aa2-b65d-cc734b01a9a3.png)

![image](https://user-images.githubusercontent.com/60442877/235332196-49660cca-6d8e-49cb-b122-dd1f281536b2.png)

![image](https://user-images.githubusercontent.com/60442877/235332715-7bd30ce7-bd83-4a03-9c57-b71288842e94.png)
