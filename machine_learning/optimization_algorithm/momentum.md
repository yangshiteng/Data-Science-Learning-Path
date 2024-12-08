![image](https://github.com/user-attachments/assets/40a2829d-d83e-4d5c-bc93-8f01546a4a96)

![image](https://github.com/user-attachments/assets/52c6d26b-fc55-4534-b1a9-e2045c8430af)

![image](https://github.com/user-attachments/assets/8d2e470c-f1fe-4c68-88c8-5c56d016c07d)

![image](https://github.com/user-attachments/assets/df09ed94-a239-4265-bc4a-6df894a67f66)

![image](https://github.com/user-attachments/assets/cd89a579-2bd8-49e3-9eba-411f23f1c1c5)

![image](https://github.com/user-attachments/assets/c3813843-9690-4cfa-a42f-42938942666d)

![image](https://github.com/user-attachments/assets/7fd7d14b-0818-4041-bd9d-23695ecdb3d9)

```python
import numpy as np

def momentum(gradient_func, initial_params, learning_rate=0.1, beta=0.9, iterations=10):
    """
    Implements the Momentum optimization algorithm.
    
    Args:
    - gradient_func: Function to compute gradients.
    - initial_params: Initial parameter values.
    - learning_rate: Learning rate.
    - beta: Momentum term (decay rate).
    - iterations: Number of iterations.
    
    Returns:
    - params_history: List of parameter values over iterations.
    """
    params = np.array(initial_params, dtype=np.float64)
    velocity = np.zeros_like(params)  # Initialize velocity
    params_history = [params.copy()]
    
    for i in range(iterations):
        # Compute gradients
        gradients = gradient_func(params)
        # Update velocity
        velocity = beta * velocity + (1 - beta) * gradients
        # Update parameters
        params -= learning_rate * velocity
        params_history.append(params.copy())
        
        print(f"Iteration {i + 1}: Params = {params}, Gradients = {gradients}, Velocity = {velocity}")
    
    return params_history

# Example: Minimize L(θ) = θ1^2 + 2θ2^2
def gradients(params):
    return np.array([2 * params[0], 4 * params[1]])

# Initial parameters
initial_params = [1.0, 2.0]

# Run Momentum
params_history = momentum(gradient_func=gradients, initial_params=initial_params, iterations=5)

# Final optimized parameters
print(f"Optimized Parameters: {params_history[-1]}")
```
![image](https://github.com/user-attachments/assets/40e876d9-cde2-472b-91ba-43fd4c1a2d2d)

![image](https://github.com/user-attachments/assets/2ac43cad-e553-4a99-a2b3-ffec8cc04e3d)

![image](https://github.com/user-attachments/assets/abc65921-d94b-49d4-bac1-1ae7f20171f0)
