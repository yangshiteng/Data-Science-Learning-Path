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

![image](https://github.com/user-attachments/assets/73ea0035-65bd-406a-9218-f5616b63b645)

![image](https://github.com/user-attachments/assets/19467884-db1c-4c1f-9382-eb08663162fd)

![image](https://github.com/user-attachments/assets/e7ea1d20-2005-41eb-94cd-783c9b258a3f)

![image](https://github.com/user-attachments/assets/9f4820ca-1993-4363-91e1-a352bdb2ff8c)

![image](https://github.com/user-attachments/assets/741805a9-5a96-41cd-8b40-47dcfd1c3b50)

```python
import numpy as np

def rmsprop(gradient_func, initial_params, learning_rate=0.1, gamma=0.9, epsilon=1e-8, iterations=10):
    """
    Implements the RMSProp optimization algorithm.
    
    Args:
    - gradient_func: Function to compute gradients.
    - initial_params: Initial parameter values.
    - learning_rate: Learning rate.
    - gamma: Decay rate for squared gradients.
    - epsilon: Small constant to prevent division by zero.
    - iterations: Number of iterations.
    
    Returns:
    - params_history: List of parameter values over iterations.
    """
    params = np.array(initial_params, dtype=np.float64)
    Eg2 = np.zeros_like(params)  # Moving average of squared gradients
    params_history = [params.copy()]
    
    for i in range(iterations):
        # Compute gradients
        gradients = gradient_func(params)
        # Update moving average of squared gradients
        Eg2 = gamma * Eg2 + (1 - gamma) * gradients**2
        # Update parameters
        params -= (learning_rate / (np.sqrt(Eg2) + epsilon)) * gradients
        params_history.append(params.copy())
        
        print(f"Iteration {i + 1}: Params = {params}, Gradients = {gradients}, Eg2 = {Eg2}")
    
    return params_history

# Example: Minimize L(θ) = θ1^2 + 2θ2^2
def gradients(params):
    return np.array([2 * params[0], 4 * params[1]])

# Initial parameters
initial_params = [1.0, 2.0]

# Run RMSProp
params_history = rmsprop(gradient_func=gradients, initial_params=initial_params, iterations=5)

# Final optimized parameters
print(f"Optimized Parameters: {params_history[-1]}")

```
![image](https://github.com/user-attachments/assets/b86ea6e5-8d30-4a85-a777-0a2599179ad7)

![image](https://github.com/user-attachments/assets/91033acd-6f78-44dd-8097-cbd80c9cd9f6)

# Adam (Adaptive Moment Estimation)

![image](https://github.com/user-attachments/assets/e87ab760-d4fe-4b89-85aa-f8e2d566e37a)

![image](https://github.com/user-attachments/assets/ec41ae65-1c1b-4555-a996-e13929ef8256)

![image](https://github.com/user-attachments/assets/85544829-135c-474f-b14f-1fb5d6bed172)

![image](https://github.com/user-attachments/assets/2e0c58d9-83d1-4739-8852-69010661d63a)

![image](https://github.com/user-attachments/assets/50bda2bd-8770-4d21-a19d-b7ace4596348)

![image](https://github.com/user-attachments/assets/5a9456b5-17c7-4a6a-8622-d4beac81ca9b)

```python
import numpy as np

def adam(gradient_func, initial_params, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, iterations=10):
    """
    Implements the Adam optimization algorithm.
    
    Args:
    - gradient_func: Function to compute gradients.
    - initial_params: Initial parameter values.
    - learning_rate: Learning rate.
    - beta1: Decay rate for first moment.
    - beta2: Decay rate for second moment.
    - epsilon: Small constant to prevent division by zero.
    - iterations: Number of iterations.
    
    Returns:
    - params_history: List of parameter values over iterations.
    """
    params = np.array(initial_params, dtype=np.float64)
    m = np.zeros_like(params)  # First moment
    v = np.zeros_like(params)  # Second moment
    params_history = [params.copy()]
    
    for t in range(1, iterations + 1):
        # Compute gradients
        gradients = gradient_func(params)
        # Update first moment
        m = beta1 * m + (1 - beta1) * gradients
        # Update second moment
        v = beta2 * v + (1 - beta2) * (gradients**2)
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        # Update parameters
        params -= (learning_rate / (np.sqrt(v_hat) + epsilon)) * m_hat
        params_history.append(params.copy())
        
        print(f"Iteration {t}: Params = {params}, Gradients = {gradients}, m = {m}, v = {v}")
    
    return params_history

# Example: Minimize L(θ) = θ1^2 + 2θ2^2
def gradients(params):
    return np.array([2 * params[0], 4 * params[1]])

# Initial parameters
initial_params = [1.0, 2.0]

# Run Adam
params_history = adam(gradient_func=gradients, initial_params=initial_params, iterations=5)

# Final optimized parameters
print(f"Optimized Parameters: {params_history[-1]}")
```

![image](https://github.com/user-attachments/assets/3dcfe6d8-2736-45f9-acb2-6d5c3f14728d)

![image](https://github.com/user-attachments/assets/1b0c0ffe-2bbc-4f13-afa6-c0898904a6d1)

![image](https://github.com/user-attachments/assets/adb4d76a-12b0-4525-a270-a950ee18f8f4)

![image](https://github.com/user-attachments/assets/58d05a2c-9f23-422f-8b03-7bee63ad78f6)

# AdaGrad vs RMSProp vs Adam

![image](https://github.com/user-attachments/assets/c3a9abfe-cccc-4115-b7dd-746fee280bba)

![image](https://github.com/user-attachments/assets/6bb3d41f-b21e-4979-a54c-2740b92f105d)

![image](https://github.com/user-attachments/assets/2bef910d-7de9-4d1c-9242-1bdb870dba8e)






