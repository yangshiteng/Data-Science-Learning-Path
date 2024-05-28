![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/4ac5889b-f343-44ff-ae87-1b6b4e2e2d86)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/088ca06e-f148-4dbd-9786-27861a3e50d5)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9d9334be-aae1-40c2-a827-736ef35628c8)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e52b3d5e-a775-4fbc-bdfa-de68abc83240)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7fd8636b-0b83-4927-9510-ce44ed3e0596)

```python
# Given probabilities
P_T_given_D = 0.99  # Probability of testing positive given the disease
P_T_given_not_D = 0.05  # Probability of testing positive given no disease

# Initial prior probability of having the disease
P_D = 0.01

# Number of tests
num_tests = 3

# Store prior and posterior probabilities for each test
priors = [P_D]
posteriors = []

for _ in range(num_tests):
    P_not_D = 1 - P_D  # Prior probability of not having the disease
    
    # Marginal probability of testing positive
    P_T = P_T_given_D * P_D + P_T_given_not_D * P_not_D
    
    # Posterior probability using Bayes' Theorem
    P_D_given_T = (P_T_given_D * P_D) / P_T
    
    # Store the results
    posteriors.append(P_D_given_T)
    priors.append(P_D_given_T)
    
    # Update prior for next iteration
    P_D = P_D_given_T

# Print the results
for i in range(num_tests):
    print(f"Test {i+1}:")
    print(f"  Prior probability: {priors[i]:.4f}")
    print(f"  Posterior probability: {posteriors[i]:.4f}")

print(f"Final posterior probability after {num_tests} positive tests: {posteriors[-1]:.4f}")
```
![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/c2816cc6-75cc-4f9d-a685-2c150eb298a9)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/e2c02055-4b23-4efb-aa29-0f0e09e5dad1)







