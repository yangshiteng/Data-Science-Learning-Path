# One-Sample Population Mean Test

## 1. One-Sample Z-Test

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/890a575b-a526-446d-8e47-31016fcf0db0)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d09bb9fc-99be-4f8f-a923-5be5a9f6d414)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/75d84035-4577-46b5-a94c-97436d152dce)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/60084af4-b9ca-4987-809f-fcc1641c23db)

    import numpy as np
    from statsmodels.stats.weightstats import ztest
    
    # Given data
    sample_mean = 78
    population_mean = 75
    population_std = 10
    sample_size = 50
    
    # Generating a sample data (assuming normal distribution for illustration)
    np.random.seed(0)  # for reproducibility
    sample_data = np.random.normal(loc=sample_mean, scale=population_std, size=sample_size)
    
    # Performing the Z-test
    z_statistic, p_value = ztest(sample_data, value=population_mean)
    
    z_statistic, p_value

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a37a5750-b18f-4d24-bc6c-af17806ccc86)

## 2. One-Sample t-test

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4720f8a3-0ea7-4e61-904a-d41b4aa81cfd)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ef25a44c-1666-41db-ab9f-947bb402e19a)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b1afeb15-786d-4148-b889-ce6bf1565a0b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9ecf5e9d-8ba3-43e4-9765-3e4f6ebabc99)

    import numpy as np
    from scipy import stats
    
    # Given values
    sample_mean = 78
    sample_std = 10
    sample_size = 20
    population_mean = 75
    
    # Simulating sample data (assuming normal distribution)
    np.random.seed(0)  # for reproducibility
    sample_data = np.random.normal(loc=sample_mean, scale=sample_std, size=sample_size)
    
    # Calculate the t-score
    t_score, p_value = stats.ttest_1samp(sample_data, population_mean)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f1f94a37-3e18-4dad-b0b1-42304d73b31d)

## 3. When Sample Size is Greater than 30

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ddba6637-402e-4588-b8c6-97d97c519e16)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7df3018b-f6ec-4471-aa01-08eb34cf270e)

# Two-Sample Population Mean Test 

## 1. Two Sample Z-test (unpaired data)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/73083d59-e2ed-43d7-b7fa-e819fa463c00)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/09a6d042-0d85-4c96-a8cf-03492f297e14)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/49308c55-3708-4f0b-b29f-f2b1290843fc)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9736140c-1b47-4561-bda7-b4e8bc668f01)

## 2. Two Sample Student's t-test (unpaired data)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/138096e1-cae4-4ab5-8edd-8c2427472e23)

## 3. Two Sample Student's t-test (paired data)

![image](https://user-images.githubusercontent.com/60442877/190540371-f8fbb1d3-5a13-4ffa-bb3e-200678f04477.png)

# One-Way ANOVA (Analysis of Variance)

Tests whether the population means of two or more independent samples are significantly different.

* H0: the means of the samples are equal.
* H1: at least one of the means of the samples are unequal

![image](https://user-images.githubusercontent.com/60442877/190549509-a7a1ae76-5a9c-4f73-a828-46e29296f813.png)

![image](https://user-images.githubusercontent.com/60442877/190549814-bd58b117-08f3-437c-adb2-108991c21508.png)

![image](https://user-images.githubusercontent.com/60442877/190550848-4be40b57-05c6-4d44-98ad-031a3f467886.png)

![image](https://user-images.githubusercontent.com/60442877/190550890-f9ee2528-8691-4278-bdc7-2c8c8775c5c8.png)

![image](https://user-images.githubusercontent.com/60442877/190551024-9f2840c7-f3e5-48e2-b2d4-da4a8ae2882c.png)

![image](https://user-images.githubusercontent.com/60442877/190551201-55f801f3-8b50-43e9-b2d9-0fc53c5866db.png)

![image](https://user-images.githubusercontent.com/60442877/190551223-16c1c253-9ef0-4dc4-a681-fe953f8c50d5.png)

![image](https://user-images.githubusercontent.com/60442877/190551315-fcd9a878-eca7-4120-acdb-e25c6c769758.png)

![image](https://user-images.githubusercontent.com/60442877/190551538-ea9274a8-324a-4f30-b193-ba3d01cf9164.png)

![image](https://user-images.githubusercontent.com/60442877/190551605-1816f83c-7a20-43ae-a232-9b076d514ece.png)

# Two-way ANOVA

![image](https://user-images.githubusercontent.com/60442877/190551757-7c9d6d9e-cdfe-4a00-adc1-9bb3c6ff3b86.png)

# Repeated Measures ANOVA Test

Tests whether the population means of two or more paired samples are significantly different.

* H0: the means of the samples are equal.
* H1: at least one of the means of the samples are unequal
