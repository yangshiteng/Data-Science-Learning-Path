![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/2eb0a29b-5744-4ed5-a1dc-e4d21e52f9b1)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c960db93-4776-49fb-a557-69b1ba2aa1f0)

# 1. Shapiro-Wilk Test for Normality

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

# 2. Kolmogorov-Smirnov Test (K-S Test) for Normality (can also test for any other distribution)

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



