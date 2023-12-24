# Pearson Correlation Coefficient Test

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/aeace57a-53f3-4880-aa07-8a2615ec7a26)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/93844427-c480-43f0-af79-01526b37b800)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0a0ee3cd-39c2-4c17-98ee-63648ca28fa7)

    import numpy as np
    import scipy.stats as stats
    
    # Generating two random sets of continuous data
    np.random.seed(0) # For reproducibility
    data1 = np.random.randn(100)  # 100 random numbers
    data2 = np.random.randn(100)  # Another set of 100 random numbers
    
    # Calculating Pearson Correlation Coefficient and p-value
    r, p_value = stats.pearsonr(data1, data2)
    
    print(f"Pearson Correlation Coefficient: {r}")
    print(f"P-value: {p_value}")

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/58226417-d9fe-44aa-835d-d669474165ad)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d80f7194-5ee4-4b91-90f5-a9aa03edac63)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/749425a4-6bdb-4280-8443-bcc8353663a0)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/8ce1ca64-5487-47dc-a2c6-e63d8dbc00ef)


# Spearman's Rank Correlation Coefficient

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3fba314e-dae9-404d-ac8f-e0b23263788d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2be791f2-07d0-4d74-84a0-c8d9aca60640)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2703b44b-084a-4f82-90ee-11aceda94a04)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/40883023-1efa-4d52-8d06-7f8942681144)

    import pandas as pd
    from scipy.stats import spearmanr
    
    # Example data
    X = [10, 20, 30, 40, 50]
    Y = [5, 15, 25, 35, 45]
    
    # Calculate Spearman's rank correlation
    rho, p_value = spearmanr(X, Y)
    
    print(f"Spearman's rank correlation coefficient: {rho}")
    print(f"P-value: {p_value}")

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0fcc858c-97bf-4bbb-806f-bdd931f98c94)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0655ddaa-3134-4f59-be28-3a0c9b717fa9)

# Difference

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ece258cc-e327-4bba-9ae0-6f78270abc96)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/828872f5-b13d-460e-9d6d-9534e9d80067)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/859afda1-da53-4a2b-ae14-36e55f3b07d2)


# Kendall's Rank Correlation

![image](https://user-images.githubusercontent.com/60442877/190044532-237da061-d455-4217-911e-6e50b9f2fd87.png)

# Pearson's Chi-Squared Test (check the independence of two categorical variables)

![image](https://user-images.githubusercontent.com/60442877/191394651-fb99e837-0176-4fc7-bf15-5681c1e442dd.png)

![image](https://user-images.githubusercontent.com/60442877/191394723-52d73d8f-3810-4e48-8b4e-d1af5c1bb6d8.png)

![image](https://user-images.githubusercontent.com/60442877/191396141-8ee0d13e-79c7-4891-b8d1-facc684c51f1.png)

![image](https://user-images.githubusercontent.com/60442877/191396831-e3006b8e-35a3-472e-879e-c98e854a0632.png)

![image](https://user-images.githubusercontent.com/60442877/191396853-669d6729-fa86-4223-b743-8cae0f20a215.png)

![image](https://user-images.githubusercontent.com/60442877/191396960-9c1c48c8-c99f-4d26-9fa8-fba75a3e424b.png)

# Fisher's Exact Test

![image](https://user-images.githubusercontent.com/60442877/191397877-26544621-e70f-4d97-91fb-e26e371d0293.png)
