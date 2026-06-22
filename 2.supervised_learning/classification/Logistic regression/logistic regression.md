In linear regression, the independent variables are used to estimate the specific value of the dependent variable.
In simple terms, it is used to create a line of **best fit** to the data to predict the values and the output of this function can be any value from $-\infty$ to $\infty$.

On the other hand, in logistic regression, the dependent variable (target) is binary (0 or 1),  so instead of predicting the values directly, logistic regression estimates the probability that outcome 1 occurs.
to turn the dependent variable from infinite values into a set of values ranging between 0 and 1. this is done by squeezing the target using a sigmoid function as shown in the figure below.
![](../Logistic%20regression/assets/images.png)

This will be explained mathematically in detail later.
### Why are we calling a classification model 'Logistic Regression'?
The reason behind this is that just like Linear Regression, logistic regression starts from a linear equation. However, this equation consists of log-odds which is further passed through a sigmoid function which squeezes the output of the linear equation to a probability between 0 and 1. And, we can decide a decision boundary and use this probability to conduct the classification task.

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

$$log(\frac{\sigma(x)}{1-\sigma(x)}) = z$$

_(Note: We use $z$ here instead of $y$, because $y$ usually represents the actual target_)
Taking the exponential to get back to odds:

$$\frac{\sigma(x)}{1-\sigma(x)} = e^{z}$$

Solving for $\sigma(x)$, we get the **sigmoid function**, which helps ensure the predicted value stays between 0 and 1:

$$\sigma(x) = \frac{e^z}{1+e^z} = \frac{1}{1+e^{-z}}$$

**The Linear Equation:**

$$f(x) = z = \theta_0 + \theta_1x_1 + ... + \theta_ix_i$$

Instead of using the linear function as it is, we vectorize it to calculate everything in one single shot:

$$f(x) = X \cdot \theta$$

Reminder:

$$X = \begin{bmatrix} 1 & x_1^{(1)} & \cdots & x_i^{(1)} \\ 1 & x_1^{(2)} & \cdots & x_i^{(2)} \\ \vdots & \vdots & & \vdots\\ 1 & x_1^{(m)} & \cdots & x_i^{(m)} \end{bmatrix} , \quad\quad \theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \vdots \\ \theta_i \end{bmatrix}, \quad\quad y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \end{bmatrix}$$

_(Vectorization is explained in detail in the Multiple Linear Regression article.)_

In linear regression, the model predicts values between $(-\infty, \infty)$. But in logistic regression, the model predicts the probability that a specific outcome occurs (0, 1). 

![](../Logistic%20regression/assets/linear-regression-vs-logistic-regression-3.webp)

That's where the sigmoid function comes in:

$$\sigma (f(x)) = \frac{1}{1+e^{-f(x)}} = \frac{1}{1 + e^{- X \cdot \theta}}$$

code implementation:
```
def sigmoid(z): 
	return 1 / (1 + np.exp(-z))
```

This transformation allows logistic regression to output the probability that the target variable      y is 1.

---
### MSE (fails)
now that we found a general way to turning the infinite target into a binary we want to find the cost and update functions just like we did in linear regression.
the process is quiet the same but with the output of the sigmoid function $\sigma(f(x))$ instead of the f(x) as we have discussed earlier.
to decide the error we take the squared error:

$$e_j^2 = (\sigma(f_\theta(x)) - y)^2$$

so the **cost** function becomes:

$$j(\theta) = \frac{1}{m}\sum_{i=1}^{m} (\sigma(f_\theta(x^i)) - y^i)^2$$

the next step is to find the update function to update the parameters of the equation.
this is done by applying the gradient descent.

$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}j(\theta)$$

by substituting the $j(\theta)$ the update function becomes:

$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}\frac{1}{m}\sum_{i=1}^{m} (\sigma(f_\theta(x^i)) - y^i)^2$$

now taking the partial derivative. the chain rule is applied here.

$$\theta_j := \theta_j-2\frac{\alpha}{m}\sum^m_{i=1}{\underbrace{\sigma(f_\theta(x^i)) - y^i}_{error}} . \underbrace{\sigma(x) .(1-\sigma(x))}_{\text{derivative of the sigmoid function}} . \underbrace{x}_{\text{derivative of the hypothesis function}}$$

While this is mathematically a valid way to calculate _an_ error, **we cannot use MSE for Logistic Regression.** here is why MSE Fails Here
When you nest the non-linear, S-shaped Sigmoid function ($\sigma$) inside a squared error function, the resulting cost function graph becomes **non-convex**.

Instead of a smooth, bowl-shaped valley with a single global minimum (like you had in Linear Regression), the loss landscape becomes wavy, filled with dozens of local minima, peaks, and flat plateaus.

If you try to run your Gradient Descent update function on this landscape, your weights will get trapped in a local minimum almost immediately, resulting in terrible predictions.

---
### MLE (valid)
Instead, we define our cost function using **Maximum Likelihood Estimation (MLE)**, and then we use **Gradient Descent** to minimize that cost function iteratively.

first we need to understand what likelihood means and what is the difference between it and probability.
so the sigmoid function outputs the probability of the output being (0 or 1).
for example if the model says that the object is 80% a satellite.
- If the image **is** actually a satellite ($y=1$), how well did our model do? It did great! The "likelihood" of this observation being correct is `0.80`.
- But what if the image is actually space debris ($y=0$), and our model predicted a `0.80` chance of it being a satellite? Then our model did terribly. The chance that our model was correct is the opposite: $1 - 0.80 =$ `0.20` (20%).

so likelihood in the first case is that the model is 80% likely to predict correctly and in the second case 20% likely to predict correctly.

however instead of using multiple if\else statements we want a single mathematical equation to do the job for us and this brings us to the **likelihood function**:

$$\text{Likelihood} = (\text{Prediction})^y \cdot (1 - \text{Prediction})^{(1 - y)}$$

now applying this function for all samples to calculate the total likelihood of the model we multiply all the likelihoods.
so the **likelihood function** becomes:

$$L(\theta) = \prod_{i=1}^{m} \sigma(z_i)^{y_i} \cdot (1-\sigma(z_i))^{(1-y_i)}$$

but If you have $1,000$ samples and each has a probability of $0.5$, you are multiplying $0.5$ by itself $1,000$ times. The result is a number so small ($0.000...$) that a computer cannot store it accurately (this is called **underflow**).
By taking the **Natural Log ($\ln$)**, we use the math property $\ln(a \cdot b) = \ln(a) + \ln(b)$. This turns the impossible multiplication into a simple **addition**.
The **Log Likelihood function** becomes:

$$\ell(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1-y^{(i)}) \log(1-\sigma(z^{(i)})) \right]$$

**MLE** wants to **maximize** this likelihood. **Gradient Descent** wants to **minimize** a cost. Therefore, we take the **Negative** average to turn it into our **Cost Function** $J(\theta)$:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1-y^{(i)}) \log(1-\sigma(z^{(i)})) \right]$$

now all we have to do is substitute this cost function into the gradient descent.
**the update function:**

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} \left( -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right] \right)$$

taking the partial derivative the update function becomes:

$$\theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^{m} \left(\sigma(z^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

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