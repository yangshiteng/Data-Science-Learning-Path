# Introduction

![image](https://github.com/user-attachments/assets/7e32cfcd-b8db-4bf1-9056-ecc7d88ff577)

![image](https://github.com/user-attachments/assets/4927f7c0-0dd5-4cf2-8f29-53d67b5456f4)

# ANOVA F-test

![image](https://github.com/user-attachments/assets/089816ad-4aed-4051-8e86-f18dd3062ce8)

![image](https://github.com/user-attachments/assets/712c01ac-d60f-4c4b-948c-fcba5dc084b4)

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Sample data
data = {
    'Feature1': [100, 200, 300, 400, 500],
    'Feature2': [10, 20, 15, 25, 30],
    'Target': [0, 1, 0, 1, 0]  # Categorical target variable
}
df = pd.DataFrame(data)

# Features and Target
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Apply SelectKBest class to extract top 2 best features using ANOVA F-Test
bestfeatures = SelectKBest(score_func=f_classif, k='all')
fit = bestfeatures.fit(X, y)

# Get scores
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

# Concatenate dataframes for better visualization 
featureScores = pd.concat([df_columns, df_scores], axis=1)
featureScores.columns = ['Feature', 'Score']  # Naming the dataframe columns

print(featureScores)
```
![image](https://github.com/user-attachments/assets/981aaa7d-30c3-4bb1-820b-37f69c2d4e3f)

# Chi-squared Test

![image](https://github.com/user-attachments/assets/1df80080-b9d8-415d-ae2f-e06e69c69252)

![image](https://github.com/user-attachments/assets/6247e7c2-365f-41f3-b534-12a7031a88d2)

```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Sample data
data = {'Feature': ['Red', 'Red', 'Green', 'Green', 'Blue', 'Blue'],
        'Target': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'Count': [10, 30, 20, 20, 30, 10]}

df = pd.DataFrame(data)

# Creating a contingency table
contingency_table = pd.pivot_table(df, values='Count', index='Feature', columns='Target', aggfunc=np.sum)

# Calculate the Chi-squared test statistic, p-value, degrees of freedom, and expected frequencies
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Print results
print("Chi-squared Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)
```
![image](https://github.com/user-attachments/assets/c5ccfac4-a848-4e4e-80bc-34c3b4fc25d4)













