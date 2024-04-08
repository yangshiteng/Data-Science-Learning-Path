![image](https://user-images.githubusercontent.com/60442877/235328976-10da73ea-11f6-47de-981c-f97e76c05e2e.png)

    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    # Load the dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Create the Random Forest classifier with OOB score enabled
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, random_state=42)

    # Train the model
    rf.fit(X, y)

    # Print the OOB score
    print("OOB score:", rf.oob_score_)

![image](https://user-images.githubusercontent.com/60442877/235328985-2deac58d-1bcf-4db3-9790-6102f5150247.png)
