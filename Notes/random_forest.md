![image](https://user-images.githubusercontent.com/60442877/235328661-24ce4b14-0d04-4d96-b028-6d970f223b48.png)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

    # Train the model
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

![image](https://user-images.githubusercontent.com/60442877/235328665-5b6ae976-160d-4bb0-9c80-88cffa613579.png)

![image](https://user-images.githubusercontent.com/60442877/235328677-92ec03ef-a1a6-4ae7-81a9-b95bb586de52.png)
