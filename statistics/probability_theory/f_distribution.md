![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/bdbba4f3-7046-4e91-b2ec-2ef059827be7)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/75efe9ec-2ecb-4ab7-a04a-5b92d5a951be)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/01296a3e-a837-43a8-ada2-7e5fa2c48b6e)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/1dd02b16-d814-4243-9308-0b169ce31b3f)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/013e68dd-74d9-4676-9f95-a8e26b089d1f)

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Sample data: Three groups
data = {
    'Group': np.repeat(['A', 'B', 'C'], 10),
    'Value': np.concatenate([np.random.normal(5, 2, 10), np.random.normal(6, 2, 10), np.random.normal(7, 2, 10)])
}
df = pd.DataFrame(data)

# Perform ANOVA
model = ols('Value ~ C(Group)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/adc15be7-cf4c-460d-a369-dbf0c85aa131)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/81d23179-703b-4902-8e8d-795e61b924a3)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/fefb7908-bd21-4b4d-87f2-8fc96d9b99a6)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/45ef63a4-5369-4a9d-8673-7b8c58262d01)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e0483243-a600-4cff-88f0-e2b915505b7c)

```python
import scipy.stats as stats

# Sample data
n1 = 10
s1_squared = 4
n2 = 12
s2_squared = 2

# Calculate the test statistic
F = s1_squared / s2_squared
print(f"F-statistic: {F}")

# Degrees of freedom
dfn = n1 - 1  # degrees of freedom for the numerator
dfd = n2 - 1  # degrees of freedom for the denominator

# Significance level
alpha = 0.05

# Critical values for a two-tailed test
F_critical_low = stats.f.ppf(alpha / 2, dfn, dfd)
F_critical_high = stats.f.ppf(1 - alpha / 2, dfn, dfd)

print(f"Lower critical value: {F_critical_low}")
print(f"Upper critical value: {F_critical_high}")

# Decision
if F < F_critical_low or F > F_critical_high:
    print("Reject the null hypothesis. The variances are significantly different.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference between the variances.")

```

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/21fac778-da6a-4205-b2b2-83f399ff9910)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b4d87453-a1a9-412b-ba2f-7b791641f478)









