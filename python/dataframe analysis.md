# 1. df.head() or df.tail()

    df.head() # first 5 rows
    df.head(10)
    
    df.tail() # last 5 rows
    df.tail(10)

# 2. df.shape

![image](https://user-images.githubusercontent.com/60442877/232264414-4be710dc-a9aa-4559-962b-c623995f8133.png)

# 3. df.info()

![image](https://user-images.githubusercontent.com/60442877/232264445-024c752c-faf9-4ff5-a947-a69794a0bcca.png)

![image](https://user-images.githubusercontent.com/60442877/232264502-d11f1b31-eb11-4d23-9081-cbe89068251e.png)

# 4. df.describe()

![image](https://user-images.githubusercontent.com/60442877/232264812-772cbbe6-5458-4287-a4b8-fd962578550b.png)

![image](https://user-images.githubusercontent.com/60442877/232264817-d9dbdf29-1059-4095-a0ad-196a2c747f52.png)

# 5. df.dtype

![image](https://user-images.githubusercontent.com/60442877/232264879-7f50f099-f6b1-4fc7-8feb-dc0a29ce5fc4.png)

![image](https://user-images.githubusercontent.com/60442877/232264937-b3955a16-ca9d-417c-b62e-01197fb6fb76.png)

# 6. df.column



![image](https://user-images.githubusercontent.com/60442877/231036438-2a2e1e6e-f2d0-4288-a3da-061fc9c9c323.png)

![image](https://user-images.githubusercontent.com/60442877/231036469-373253a4-84a8-4137-bc27-74be4624aae3.png)

    import pandas as pd
    import numpy as np

    # Create a sample DataFrame
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, np.nan],
        'B': [5, 6, 7, 8, 9],
        'C': ['apple', 'banana', 'orange', 'grape', 'pear']
    })

    print("Sample DataFrame:")
    print(data)

    # head()
    print("\nFirst 3 rows using head():")
    print(data.head(3))

    # tail()
    print("\nLast 3 rows using tail():")
    print(data.tail(3))

    # shape
    print("\nShape (rows, columns) using shape:")
    print(data.shape)

    # info()
    print("\nDataFrame summary using info():")
    print(data.info())

    # describe()
    print("\nSummary statistics of numerical columns using describe():")
    print(data.describe())

    # dtypes
    print("\nData types of columns using dtypes:")
    print(data.dtypes)

    # columns
    print("\nColumn names using columns:")
    print(data.columns)

    # index
    print("\nIndex using index:")
    print(data.index)

    # value_counts()
    print("\nValue counts for column 'C' using value_counts():")
    print(data['C'].value_counts())

    # isnull()
    print("\nCheck for null values using isnull():")
    print(data.isnull())

    # notnull()
    print("\nCheck for non-null values using notnull():")
    print(data.notnull())

    # corr()
    print("\nPairwise correlation of numerical columns using corr():")
    print(data.corr())

![image](https://user-images.githubusercontent.com/60442877/231044390-958449b0-0b47-4748-b0bd-6006e7ebd7e0.png)

![image](https://user-images.githubusercontent.com/60442877/231044480-2bf367d3-316f-4f0c-a6cc-ef65dab17f4b.png)

![image](https://user-images.githubusercontent.com/60442877/231044509-1b1dd2f1-079a-4d2e-bc59-9e47f64b2da9.png)


