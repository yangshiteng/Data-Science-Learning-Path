# Chi-Squared Test of Independence

![image](https://user-images.githubusercontent.com/60442877/192128552-fca7962f-1ef3-4282-9258-3c1d1e339f61.png)

![image](https://user-images.githubusercontent.com/60442877/192128555-762caab2-a1b0-4eaf-9606-250072beff10.png)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/89e39318-7034-46ae-8bd1-59a54e6d6ef2)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/49af71d1-bee0-4ec5-b9b7-e8486b477205)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/893d111f-7b6f-45b6-80f4-dd7e179daf86)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/3789f4e8-8dd0-4751-835e-a35f443a34fa)

    import numpy as np
    from scipy.stats import chi2_contingency
    
    # Hypothetical data: rows are Gender, columns are Preference
    data = np.array([[30, 10, 60],
                     [35, 20, 45]])
    
    # Performing the Chi-Squared test
    chi2, p, dof, expected = chi2_contingency(data)
    
    print("Chi-Squared Statistic:", chi2)
    print("Degrees of Freedom:", dof)
    print("P-value:", p)
    print("Expected Frequencies:\n", expected)
    
    # Interpretation
    alpha = 0.05
    if p < alpha:
        print("Reject the Null Hypothesis: There is a significant association between the variables.")
    else:
        print("Fail to Reject the Null Hypothesis: No significant association between the variables.")

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c204b574-d32d-474a-a022-e0a35ec8ffbb)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f5da0087-d0e8-4c89-8ec8-eec4d940ce0a)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c2c1c8aa-c2a7-4ffc-9ca4-2b226628af28)

# Fisher's Exact Test

![image](https://user-images.githubusercontent.com/60442877/192128655-d0ad68ef-91c8-416b-b98d-764616a4ab44.png)

![image](https://user-images.githubusercontent.com/60442877/192128659-7d187da0-aaa3-4a04-8879-cffb3fc3504e.png)

https://www.statology.org/fishers-exact-test-calculator/

![image](https://user-images.githubusercontent.com/60442877/192128663-d02002c7-df1d-482a-800e-996bb5c206a1.png)

![image](https://user-images.githubusercontent.com/60442877/192128680-e08f86d3-61ad-4ab7-8f35-d5aae9aeca51.png)

![image](https://user-images.githubusercontent.com/60442877/192128688-f973b20c-a055-4b63-9003-981f578e2bc8.png)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/647bdb15-9532-4558-8ba3-e1e64d9ac471)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/522ebb97-781b-4fc0-87fe-16bb52448220)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/147bd0b5-38ca-4610-98b7-84e4ac3396b3)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/a9491713-e303-416a-8a01-8f2087efd86a)

    from scipy.stats import fisher_exact
    
    # Hypothetical data in a 2x2 table
    # Format: [[Treatment and Success, Treatment and Failure], [Control and Success, Control and Failure]]
    data = [[8, 2], [1, 9]]
    
    # Performing Fisher's Exact Test
    odds_ratio, p_value = fisher_exact(data)
    
    print("Odds Ratio:", odds_ratio)
    print("P-value:", p_value)
    
    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print("Reject the Null Hypothesis: There is a significant association between the variables.")
    else:
        print("Fail to Reject the Null Hypothesis: No significant association between the variables.")

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/753bafff-54f6-4cfd-8b26-f209e2729e55)
















        
