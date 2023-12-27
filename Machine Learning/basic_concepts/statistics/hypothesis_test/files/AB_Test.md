# Concept

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ba087679-1116-4e0e-8772-d246e770ddca)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d9af14a0-92cb-4014-8f05-90cb9f251a6a)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c1e6b293-6ad1-4204-bcca-6f07819a7d5c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/ae2fa293-7de6-4ee6-95e1-08e03d1c46d6)

# Case Study

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/34bd89be-fbf4-4f52-af01-a6321c32e49d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e6573399-7952-4599-875e-104d68bd3280)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f1d1ea5b-f1d4-48fa-81a5-17eb6f88ba29)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/85a2ac7b-e0bb-4fc4-b801-abecbf11b610)

# Calculation Detail (Two Sample Proportion Z-test)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/5a256d11-6fa8-4dcd-8b2c-793fff8b4891)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b87d615a-bb8b-47ff-a3d7-da73fd9b8408)

    import numpy as np
    from scipy.stats import norm
    
    # Ad A data
    clicks_a = 500
    impressions_a = 10000
    ctr_a = clicks_a / impressions_a
    
    # Ad B data
    clicks_b = 600
    impressions_b = 10000
    ctr_b = clicks_b / impressions_b
    
    # Pooled CTR
    pooled_ctr = (clicks_a + clicks_b) / (impressions_a + impressions_b)
    
    # Z-score calculation
    z_score = (ctr_a - ctr_b) / np.sqrt(pooled_ctr * (1 - pooled_ctr) * (1/impressions_a + 1/impressions_b))
    
    # P-value
    p_value = 2 * norm.sf(np.abs(z_score)) # Two-tailed test
    
    print(f"Z-score: {z_score}")
    print(f"P-value: {p_value}")

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/80813ff1-6fac-4464-8707-d56fc7b911d4)
