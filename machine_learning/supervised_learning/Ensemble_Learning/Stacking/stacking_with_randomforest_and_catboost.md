![image](https://user-images.githubusercontent.com/60442877/235336334-d0817108-5442-42ec-a158-6264171a7016.png)

    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.metrics import accuracy_score
    from catboost import CatBoostClassifier

    # Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the base models
    base_models = [
        ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('catboost', CatBoostClassifier(iterations=100, learning_rate=0.1, depth=4, verbose=0, random_seed=42))
    ]

    # Create the stacking ensemble
    stacking_clf = StackingClassifier(estimators=base_models, final_estimator=None, cv=5)

    # Train the stacking ensemble
    stacking_clf.fit(X_train, y_train)

    # Make predictions
    y_pred = stacking_clf.predict(X_test)

    # Evaluate the ensemble
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

![image](https://user-images.githubusercontent.com/60442877/235336372-28ac9f97-5fdd-40e4-8a15-02551c776d38.png)

