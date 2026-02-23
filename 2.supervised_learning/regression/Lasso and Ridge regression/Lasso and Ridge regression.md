In multiple linear regression, the model uses several independent variables to estimate the specific value of the continuous dependent variable.

In simple terms, it tries to find the perfect weight ($\theta$) for every single feature to create a multidimensional plane of **best fit**.

But what happens when we feed the model too much information? the standard regression math panics. It assigns massive, unstable weights to the useless features, causing the model to perfectly memorize the training data but fail in the real world. This is called **Overfitting**.

To fix this, we use a technique called **Regularization**. Instead of just minimizing the error, Regularization adds a **mathematical penalty** to the cost function. It physically punishes the model for having large weights ($\theta$), forcing it to build simpler, more generalized rules.

This will be explained mathematically in detail below through the two most common regularization techniques: **Ridge (L2)** and **Lasso (L1)**.

### Why do we penalize the weights?
Think of the weights ($\theta$) as the "volume knob" for each feature. If the model is overfitting, it means some of the volume knobs are turned up way too high on noisy data. By adding a penalty to the cost function, we force the gradient descent algorithm to turn those volume knobs down.

---

## Mathematical Explanation: Ridge Regression (L2)

Before we dive into the regularized cost function, let's look at the standard Multiple Linear Regression Cost Function (Mean Squared Error) that we want to minimize:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

In **Ridge Regression**, we add a penalty term to the end of this cost function. This penalty is the **squared magnitude** of the coefficients.

The Ridge Cost Function becomes:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} \theta_j^2$$

Here, $\lambda$ (Lambda) is the **regularization parameter**. It controls how harsh the penalty is.

- If $\lambda = 0$, the penalty disappears, and you just have standard Linear Regression.
    
- If $\lambda$ is very large, the penalty is so heavy that it forces all the weights to shrink very close to zero, which could cause _underfitting_.
	

_(Note: We only sum from $j=1$ to $n$. We **do not** penalize $\theta_0$, which is the y-intercept or bias term, because shifting the line up or down doesn't cause overfitting)._

Because Ridge squares the weights ($\theta_j^2$), it punishes exceptionally large weights very heavily, but it never pushes a weight to _exactly_ zero. It just shrinks them to be very, very small.

**The Update Function for Ridge:**
Taking the partial derivative of this new cost function, our Gradient Descent update rule changes slightly to include the derivative of the penalty:

$$\theta_j := \theta_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} \theta_j \right]$$

code implementation:

```
for i in range(iterations):
    # 1. Calculate predictions
    predictions = np.dot(X_vectorized, theta)
    
    # 2. Calculate errors
    error = predictions - y
    
    # 3. Create a copy of theta and set theta_0 to zero (we don't penalize the bias!)
    theta_penalized = np.copy(theta)
    theta_penalized[0] = 0 
    
    # 4. Calculate gradient WITH the L2 penalty
    gradient = (1/m) * np.dot(X_vectorized.T, error) + (lambda_ / m) * theta_penalized
    
    # 5. Update theta
    theta = theta - (learning_rate * gradient)
```

---

## Mathematical Explanation: Lasso Regression (L1)

Ridge Regression is great, but what if we have a dataset with 1,000 features and we _know_ 900 of them are completely useless? We don't just want to shrink the useless weights; we want to delete them entirely.

This is where **Lasso Regression** (Least Absolute Shrinkage and Selection Operator) comes in. Instead of adding the squared magnitude, Lasso adds the **absolute value** of the coefficients as the penalty.

The Lasso Cost Function becomes:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} |\theta_j|$$

Because of the way the absolute value math works (which creates sharp corners in the geometric space of the function), Lasso doesn't just shrink weights—it drives the weights of 
useless features to **exactly zero**.

This means Lasso performs automatic **Feature Selection**. It looks at the "owner's shoe size" column, realizes it doesn't help predict house prices, and mathematically deletes it by making its $\theta = 0$.

**The Update Function for Lasso:**
The derivative of an absolute value $|\theta_j|$ is the `sign` of $\theta_j$ (which is +1 if $\theta$ is positive, and -1 if $\theta$ is negative).

$$\theta_j := \theta_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} \text{sign}(\theta_j) \right]$$

code implementation:

```
for i in range(iterations):
    # 1. Calculate predictions
    predictions = np.dot(X_vectorized, theta)
    
    # 2. Calculate errors
    error = predictions - y
    
    # 3. Create a copy of theta and set theta_0 to zero
    theta_penalized = np.copy(theta)
    theta_penalized[0] = 0 
    
    # 4. Calculate gradient WITH the L1 penalty using np.sign()
    gradient = (1/m) * np.dot(X_vectorized.T, error) + (lambda_ / m) * np.sign(theta_penalized)
    
    # 5. Update theta
    theta = theta - (learning_rate * gradient)
```

### Summary: Which one do we use?

- Use **Ridge (L2)** when you have many features that you believe are all somewhat useful, and you just want to prevent the model from assigning crazy high weights to any single one of them.
    
- Use **Lasso (L1)** when you have a massive amount of features and you suspect many of them are completely useless. It will act as a bouncer, throwing the bad features out of your equation entirely.

for deeper understanding make sure to check these sources  
- [he Mathematical background of Lasso and Ridge Regression   **(medium)**](https://medium.com/codex/mathematical-background-of-lasso-and-ridge-regression-23b74737c817)
- [What is lasso regression? **(IBM)**](https://www.ibm.com/think/topics/lasso-regression#1190488335)
- [What is ridge regression?   **(IBM)**](https://www.ibm.com/think/topics/ridge-regression#1190488336)