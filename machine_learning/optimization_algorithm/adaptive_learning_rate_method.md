# AdaGrad (Adaptive Gradient Algorithm)

![image](https://github.com/user-attachments/assets/f98a9f75-6a82-43b5-8265-4b6292dacf3b)

![image](https://github.com/user-attachments/assets/5c6f1f04-5f26-4229-8053-dd2cb0872777)

![image](https://github.com/user-attachments/assets/cf68baa7-4805-4092-b260-b1cbb8a10714)

![image](https://github.com/user-attachments/assets/976d95a3-4de5-497a-a33c-65d35dea4246)

![image](https://github.com/user-attachments/assets/ba34a002-7a44-4769-b080-0b93b2c15e8c)

```python
import numpy as np

def adagrad(gradient_func, initial_params, learning_rate=0.1, epsilon=1e-8, iterations=10):
    """
    Implements the AdaGrad optimization algorithm.
    
    Args:
    - gradient_func: Function to compute gradients.
    - initial_params: Initial parameter values.
    - learning_rate: Initial learning rate.
    - epsilon: Small constant to prevent division by zero.
    - iterations: Number of iterations.
    
    Returns:
    - params_history: List of parameter values over iterations.
    """
    params = np.array(initial_params, dtype=np.float64)
    G = np.zeros_like(params)  # Accumulated squared gradients
    params_history = [params.copy()]
    
    for i in range(iterations):
        # Compute gradients
        gradients = gradient_func(params)
        # Accumulate squared gradients
        G += gradients**2
        # Update parameters
        params -= (learning_rate / (np.sqrt(G) + epsilon)) * gradients
        params_history.append(params.copy())
        
        print(f"Iteration {i + 1}: Params = {params}, Gradients = {gradients}, G = {G}")
    
    return params_history

# Example: Minimize L(θ) = θ1^2 + 2θ2^2
def gradients(params):
    return np.array([2 * params[0], 4 * params[1]])

# Initial parameters
initial_params = [1.0, 2.0]

# Run AdaGrad
params_history = adagrad(gradient_func=gradients, initial_params=initial_params, iterations=5)

# Final optimized parameters
print(f"Optimized Parameters: {params_history[-1]}")

```

![image](https://github.com/user-attachments/assets/b1591238-52d3-4888-a888-2deba002e8cc)

![image](https://github.com/user-attachments/assets/99f750b2-6e03-4189-8e30-ca6c26df4644)


# RMSProp (Root Mean Square Propagation)

![image](https://github.com/user-attachments/assets/1817c579-6e9c-472f-8193-5b93c340160d)

![image](https://github.com/user-attachments/assets/fdeff1dc-4f66-44dd-9800-489aad97f2ad)

![image](https://github.com/user-attachments/assets/5238924c-bc82-42c9-87d9-8064b805e0c9)

# Adam

![image](https://github.com/user-attachments/assets/a9336442-55cf-4fcd-993d-ef584f88feb3)

![image](https://github.com/user-attachments/assets/e87075fd-599f-49f2-b578-3ead4153aea3)

![image](https://github.com/user-attachments/assets/6f30f2f7-8e80-427a-bd4b-c6956c62a617)
