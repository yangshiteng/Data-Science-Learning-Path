![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/05dfaf50-1891-4165-90e9-fa1b56e9b31b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/b7af5ccd-5d1b-4611-8317-78b4eede9ece)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/0b98a46d-369b-4bbf-ae3f-2ef7397aabec)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/1df7150d-c3c1-4a60-9fc8-f36e0ab142e2)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/70eacb6d-2596-4bb9-bcdc-7a719ce08919)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/11df02ec-1278-44dc-ad9d-68420cb7ed6b)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/eca9537f-d72f-4ae3-b0c8-5fa2f85437b0)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7652c205-3ac0-4927-8572-276414d0f67c)

### Maximum Likelihood Estimation (MLE)

**Introduction:**

Maximum Likelihood Estimation (MLE) is a statistical method used to estimate the parameters of a probability distribution by maximizing the likelihood function, which measures how well the model explains the observed data.

### Steps in MLE

1. **Specify the Probability Distribution**: Choose the probability distribution that best represents the data.
2. **Write the Likelihood Function**: Construct the likelihood function based on the chosen probability distribution.
3. **Take the Log-Likelihood**: For mathematical convenience, take the natural logarithm of the likelihood function.
4. **Differentiate the Log-Likelihood**: Differentiate the log-likelihood with respect to the parameter(s).
5. **Solve for the Parameters**: Set the derivative(s) to zero and solve for the parameter(s).
6. **Numerical Method**: If an analytical solution is not feasible, use numerical optimization techniques to maximize the log-likelihood function.

### Example: Estimating the Mean and Variance of a Normal Distribution

Let's assume we have a set of data points that we believe are normally distributed. We want to estimate the mean (\(\mu\)) and the standard deviation (\(\sigma\)) of this distribution using MLE.

#### Analytical Solution

For a normal distribution, the MLE estimators for \(\mu\) and \(\sigma^2\) are:

1. **Likelihood Function**:
   The probability density function (PDF) of a normal distribution is:
   \[
   f(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
   \]
   
   The likelihood function for \(n\) independent observations \(x_1, x_2, \ldots, x_n\) is the product of their PDFs:
   \[
   L(\mu, \sigma | x_1, x_2, \ldots, x_n) = \prod_{i=1}^n f(x_i | \mu, \sigma)
   \]

2. **Log-Likelihood Function**:
   Taking the natural logarithm of the likelihood function gives the log-likelihood function:
   \[
   \log L(\mu, \sigma | x_1, x_2, \ldots, x_n) = \sum_{i=1}^n \left( -\frac{1}{2} \log(2\pi\sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2} \right)
   \]

3. **Differentiate the Log-Likelihood**:
   Differentiate the log-likelihood function with respect to \(\mu\) and \(\sigma\):

   For \(\mu\):
   \[
   \frac{\partial \log L}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^n (x_i - \mu) = 0
   \]
   Solving for \(\mu\), we get:
   \[
   \hat{\mu} = \frac{1}{n} \sum_{i=1}^n x_i
   \]

   For \(\sigma\):
   \[
   \frac{\partial \log L}{\partial \sigma} = -\frac{n}{\sigma} + \frac{1}{\sigma^3} \sum_{i=1}^n (x_i - \mu)^2 = 0
   \]
   Solving for \(\sigma\), we get:
   \[
   \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2
   \]

Thus, the MLE for \(\mu\) is the sample mean, and the MLE for \(\sigma^2\) is the sample variance (without Bessel's correction).

#### Numerical Method

When the likelihood equations are complex and do not have closed-form solutions, we use numerical optimization techniques. Here, we'll focus on the steps and formulas involved in the numerical method.

1. **Define the Negative Log-Likelihood Function**:
   To facilitate the optimization process, we define the negative log-likelihood function. For a normal distribution, the negative log-likelihood function is:
   \[
   -\log L(\mu, \sigma | x_1, x_2, \ldots, x_n) = \frac{n}{2} \log(2\pi\sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2
   \]

2. **Initial Guess**:
   Start with an initial guess for the parameters \(\mu\) and \(\sigma\). For example, \([0, 1]\) could be used as initial guesses for the mean and standard deviation.

3. **Set Bounds**:
   Set appropriate bounds for the parameters. Ensure that \(\sigma\) is positive by setting its lower bound to a small positive number (e.g., 0.001).

4. **Optimization Process**:
   Use an optimization algorithm to find the parameter values that minimize the negative log-likelihood function. Common algorithms include gradient descent, Newton-Raphson, and quasi-Newton methods (such as BFGS). The optimization algorithm iteratively adjusts the parameter values to find the minimum of the objective function.

   For instance, in the gradient descent method, the parameters are updated iteratively as follows:
   \[
   \theta_{new} = \theta_{old} - \eta \nabla_{\theta} (-\log L(\theta))
   \]
   where \(\theta\) represents the parameters \(\mu\) and \(\sigma\), \(\eta\) is the learning rate, and \(\nabla_{\theta} (-\log L(\theta))\) is the gradient of the negative log-likelihood function with respect to the parameters.

5. **Extract Results**:
   Once the optimization process converges, extract the estimated parameters. The final values of \(\mu\) and \(\sigma\) that minimize the negative log-likelihood function are the MLE estimates.

### Summary

- **Analytical Solution**: For a normal distribution, the MLE for the mean \(\mu\) is the sample mean, and the MLE for the variance \(\sigma^2\) is the sample variance (without Bessel's correction).
- **Numerical Method**: Uses optimization techniques to minimize the negative log-likelihood function, providing MLE estimates for the parameters. This involves defining the negative log-likelihood function, choosing an initial guess, setting bounds, and iteratively updating the parameters using an optimization algorithm until convergence.

Both methods aim to find the parameter values that maximize the likelihood function. The analytical solution is straightforward for simple models like the normal distribution, while the numerical method is more versatile and can handle more complex models where closed-form solutions are not feasible.
