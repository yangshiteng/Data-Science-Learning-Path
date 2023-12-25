# Chi-Squared Test of Independence

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
