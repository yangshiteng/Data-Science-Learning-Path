![image](https://user-images.githubusercontent.com/60442877/235335919-7a10216e-eb51-4efb-af1a-6ae39878cd3c.png)

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor

    # Load the dataset
    boston = load_boston()
    X = boston.data
    y = boston.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create the base estimator (weak learner)
    base_estimator = DecisionTreeRegressor(max_depth=4)

    # Create the AdaBoost regressor
    ada = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=50, learning_rate=1, random_state=42)

    # Train the model
    ada.fit(X_train, y_train)

    # Make predictions
    y_pred = ada.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

![image](https://user-images.githubusercontent.com/60442877/235334770-60278ff9-411c-48c1-a97d-48e9efbacc80.png)
