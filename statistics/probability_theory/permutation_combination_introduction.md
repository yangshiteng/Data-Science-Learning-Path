![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/550299cd-7e1b-4d29-beaa-98c3c9ba5bb3)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/aca85456-7014-431a-ab57-77ee7195826c)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/f1be484e-d37a-4054-bd52-3f8e520b0940)

```python
import math

# Factorial
def factorial(n):
    return math.factorial(n)

# Permutation
def permutation(n, r):
    return math.factorial(n) // math.factorial(n - r)

# Combination
def combination(n, r):
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

# Examples
print("Factorial of 5:", factorial(5))
print("Permutations of 5 taken 3 at a time:", permutation(5, 3))
print("Combinations of 5 taken 3 at a time:", combination(5, 3))
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/d04411cc-4430-4697-8b92-c579e1573ec9)
