![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2eb0a29b-5744-4ed5-a1dc-e4d21e52f9b1)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c960db93-4776-49fb-a557-69b1ba2aa1f0)

# 1. Shapiro-Wilk Test for Normality (small to medium dataset)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/5c2ce78a-b8ca-404f-9ff2-af5e542f5231)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/57d4a7e5-a2e5-4f58-8af5-536479d01c72)

    import numpy as np
    from scipy import stats
    
    # Sample data
    data = np.random.normal(loc=0, scale=1, size=100)  # Generate 100 normal distributed data points
    
    # Perform Shapiro-Wilk test
    w_statistic, p_value = stats.shapiro(data)
    
    # Interpret the results
    alpha = 0.05
    if p_value > alpha:
        print(f"Fail to reject the null hypothesis (p-value = {p_value:.3f}). Data seems to be normally distributed.")
    else:
        print(f"Reject the null hypothesis (p-value = {p_value:.3f}). Data does not seem to be normally distributed.")

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4acc34ec-bebf-40e1-9470-536b4b2db9ec)

# 2. Kolmogorov-Smirnov Test (K-S Test) for Normality (can also test for any other distribution) (larger dataset)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/66ac63da-adde-4812-b1b9-a550e121dac9)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7576b0b8-fd4e-479d-b333-5b8246751da6)

    import numpy as np
    from scipy import stats
    
    # Generate a sample data
    data = np.random.normal(loc=0, scale=1, size=100)  # 100 data points from a normal distribution
    
    # Perform the Kolmogorov-Smirnov test
    d_statistic, p_value = stats.kstest(data, 'norm')
    
    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print(f"Reject the null hypothesis (p-value = {p_value:.3f}). Data does not seem to follow a normal distribution.")
    else:
        print(f"Fail to reject the null hypothesis (p-value = {p_value:.3f}). Data seems to follow a normal distribution.")

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/310a5b58-836c-46dc-aebe-324dc5d2a7ae)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4af0f950-dcbc-4390-a21c-7db5e9cc4518)

    import numpy as np
    from scipy import stats
    
    # Generate sample data from a normal distribution with mean=5 and standard deviation=3
    data = np.random.normal(loc=5, scale=3, size=100)  # loc is mean, scale is standard deviation
    
    # Perform the Kolmogorov-Smirnov test against a normal distribution with mean=5 and std=3
    d_statistic, p_value = stats.kstest(data, 'norm', args=(5, 3))
    
    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print(f"Reject the null hypothesis (p-value = {p_value:.3f}). Data does not seem to follow a normal distribution with mean 5 and std 3.")
    else:
        print(f"Fail to reject the null hypothesis (p-value = {p_value:.3f}). Data seems to follow a normal distribution with mean 5 and std 3.")

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/1ecff293-5bb5-4bdd-9a7f-137509a1e278)

# 3. Lilliefors Test for Normality (modification of the Kolmogorov-Smirnov Test) (for unknown population mean and variance)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0317d6a2-e766-41af-b134-ae77e7c0e171)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/47b16435-bacd-455b-943e-1237f0c8e59f)

    import numpy as np
    from statsmodels.stats.diagnostic import lilliefors
    
    # Sample data
    data = np.random.normal(loc=0, scale=1, size=100)  # Generate 100 data points from a normal distribution
    
    # Perform Lilliefors test
    d_statistic, p_value = lilliefors(data)
    
    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print(f"Reject the null hypothesis (p-value = {p_value:.3f}). Data does not seem to be normally distributed.")
    else:
        print(f"Fail to reject the null hypothesis (p-value = {p_value:.3f}). Data seems to be normally distributed.")

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/57824c7b-8f77-450f-a807-302bc0ad40f7)


