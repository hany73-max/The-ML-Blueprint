In linear regression, the independent variables are used to estimate the specific value of the dependent variable.
In simple terms, it is used to create a line of **best fit** to the data to predict the values.

On the other hand, in logistic regression, the dependent variable is dichotomous (0 or 1), so instead of predicting the values directly, logistic regression estimates the probability that outcome 1 occurs.

The output of the model is a value between 0 and 1. Based on a threshold (often 0.5), **then** we classify the outcome as either "approved" or "not approved".
It is a statistical model typically used to model a binary dependent variable with the help of the logistic function (sigmoid).

This will be explained mathematically in detail later.

### Why are we calling a classification model 'Logistic Regression'?
The reason behind this is that just like Linear Regression, logistic regression starts from a linear equation. However, this equation consists of log-odds which is further passed through a sigmoid function which squeezes the output of the linear equation to a probability between 0 and 1. And, we can decide a decision boundary and use this probability to conduct the classification task.

### Example: Logistic Regression Model

WIP

---

## Mathematical Explanation:

Before we dive into logistic regression, we first need to understand some concepts like probability and odds, which are briefly explained here.

For example, take a box filled with 3 orange balls and 5 red ones:

**Probability** measures the chance of an event occurring out of all possible outcomes.
- The probability of picking a red ball = $5 / (3+5) = 0.625$.

**Odds** compare the chance of an event occurring to the chance of it not occurring.
 - The odds of picking a red ball = $5 / 3 = 1.667$.

However, the odds are not symmetric around 1. For example, odds of 2 and 0.5 represent "twice as likely" and "half as likely," but they’re on very different numerical scales. To address this imbalance, we take the logarithm of the odds, which transforms the unbounded $(0, \infty)$ scale of odds to the real number line $(-\infty, \infty)$. This is known as the **log-odds**, or **logit**.

The logit is defined as:

$$log(\frac{f(x)}{1-f(x)}) = z$$

_(Note: We use $z$ here instead of $y$, because $y$ usually represents the actual label 0 or 1)_

Taking the exponential to get back to odds:

$$\frac{f(x)}{1-f(x)} = e^{z}$$

Solving for $f(x)$, we get the **sigmoid function**, which helps ensure the predicted value stays between 0 and 1:

$$f(x) = \frac{e^z}{1+e^z} = \frac{1}{1+e^{-z}}$$

**The Linear Equation:**

$$z = \theta_0 + \theta_1x_1 + ... + \theta_ix_i$$

Instead of using the linear function as it is, we vectorize it to calculate everything in one single shot:

$$z = X \cdot \theta$$

Reminder:

$$X = \begin{bmatrix} 1 & x_1^{(1)} & \cdots & x_i^{(1)} \\ 1 & x_1^{(2)} & \cdots & x_i^{(2)} \\ \vdots & \vdots & & \vdots\\ 1 & x_1^{(m)} & \cdots & x_i^{(m)} \end{bmatrix} , \quad\quad \theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_i \end{bmatrix}, \quad\quad y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \end{bmatrix}$$

_(Vectorization is explained in detail in the Multiple Linear Regression article.)_

In linear regression, the model predicts values between $(-\infty, \infty)$. But in logistic regression, the model predicts the probability that a specific outcome occurs (0, 1). 

![](../logistic%20regression/assets/linear-regression-vs-logistic-regression-3.webp)

That's where the sigmoid function comes in:

$$\sigma (f(x)) = \frac{1}{1+e^{-z}} = \frac{1}{1 + e^{- X \cdot \theta}}$$

code implementation:

```
def sigmoid(z): 
	return 1 / (1 + np.exp(-z))
```

This transformation allows logistic regression to output the probability that the target variable      y is 1.

Now that we found a general function for the logistic regression output, we want to optimize the parameters. Unlike linear regression, there is no closed-form "Normal Equation" to find the optimum $\theta$ directly.

Instead, we define our cost function using **Maximum Likelihood Estimation (MLE)**, and then we use **Gradient Descent** to minimize that cost function iteratively.

To understand the **maximum likelihood method**, we introduce the **likelihood function $L(\theta)$**.

 $L(\theta)$ indicates how likely it is that the observed data occur. As _θ_ changes, the probability of the observed data changes.
- If a data point is actually $y=1$, we want our model's prediction $f(x)$ to be as close to $1$ as possible.
- If a data point is actually $y=0$, we want our prediction to be as close to $0$ as possible (meaning $1 - f(x)$ should be large).

![](../logistic%20regression/assets/Maximum-likelihood-estimation.png)

so the goal her is to **maximize the likelihood function**.we will achieve that by multiplying the predicted probabilities for each data point.

so the **likelihood function** becomes:

$$L(\theta) = \prod_{i=1}^{m} f(x_i)^{y_i} \cdot (1-f(x_i))^{(1-y_i)}$$

but If you have $1,000$ samples and each has a probability of $0.5$, you are multiplying $0.5$ by itself $1,000$ times. The result is a number so small ($0.000...$) that a computer cannot store it accurately (this is called **underflow**).

By taking the **Natural Log ($\ln$)**, we use the math property $\ln(a \cdot b) = \ln(a) + \ln(b)$. This turns the impossible multiplication into a simple **addition**.

The **Log Likelihood function** becomes:

$$\ell(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(f(x^{(i)})) + (1-y^{(i)}) \log(1-f(x^{(i)})) \right]$$

**MLE** wants to **maximize** this likelihood. **Gradient Descent** wants to **minimize** a cost. Therefore, we take the **Negative** average to turn it into our **Cost Function** $J(\theta)$:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f(x^{(i)})) + (1-y^{(i)}) \log(1-f(x^{(i)})) \right]$$

now all we have to do is substitute this cost function into the gradient descent.

**the update function:**

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} \left( -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f(x^{(i)})) + (1 - y^{(i)}) \log(1 - f(x^{(i)})) \right] \right)$$

taking the partial derivative the update function becomes:

$$\theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^{m} \left(f(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

notice that the update function is the same as linear regression.

now for the implementation we will convert this function into a vectorized one like we did in multiple linear regression:

$$\theta := \theta - \alpha \cdot \frac{1}{m} \cdot \underbrace{X^T \cdot (\sigma (X\theta) - y)}_{\text{The Gradient}}$$

code implementation:

```
for i in range(iterations):
    # 1. Calculate predictions for all samples
    z = np.dot(X_vectorized, theta)
    
    # 2. Apply Sigmoid Activation (Predictions)
    predictions = sigmoid(z)
    
    # 3. Calculate errors
    error = predictions - y
    
    # 4. Calculate gradient
    gradient = (1/m) * np.dot(X_vectorized.T, error)
    
    # 5. Update theta
    theta = theta - (learning_rate * gradient)
```