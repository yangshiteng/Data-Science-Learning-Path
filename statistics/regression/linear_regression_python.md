![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/69b66e50-7777-465f-91c6-ac44f51be3ff)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/97e97c95-18ba-48fc-a2a3-2c0f318d8619)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/93af5ccd-1299-4bfc-8e73-4e46685ff251)

    # Example dataset
    X = np.array([[1], [2], [3], [4], [5]])  # Independent variable(s)
    y = np.array([2, 4, 5, 4, 5])  # Dependent variable
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/6540f66a-494a-4c3c-af7a-1da9a42c0641)

    model = LinearRegression()

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0e758158-7eda-430b-ac47-c6cd004cc3a8)

    model.fit(X_train, y_train)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/61acf6ef-3d78-4792-b3c1-f641f0597a37)

    y_pred = model.predict(X_test)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/5ca30b26-e814-43f3-b6bc-f981f837ad78)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/37ceff6e-be83-4033-ad0d-e6cd98cfb2dc)

    plt.scatter(X_test, y_test, color='black', label='Actual data')
    plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Fitted line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/dc4f38e4-c296-4efe-a708-708393e85050)

    # Coefficients and Intercept
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ec111094-de0c-4381-ab93-7211128a5a8c)

