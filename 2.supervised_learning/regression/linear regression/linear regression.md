Linear Regression is a statistical tool that produces a line of best fit for a given dataset analytically. To produce the regression line manually, one needs to perform operations such as mean-squared error and optimizing the cost function; both are explained in detail later.
### Example: Linear Regression Model
![](../linear%20regression/assets/Screenshot%202026-01-20%20210035.png)
## mathematical explanation:
the line in the figure can be described using the linear equation:

$$f(x) = x_1\theta_1 + \theta_0$$

##### in code:
```
y_pred = np.dot(x, weight) + bias
```

the line we drew to describe the data might not fit perfectly from the very first try so what we do is 
measure the error of the line by calculating the difference between the output of the line equation and the actual data:

$$error = e_j = f_\theta(x)-y$$

to remove any negatives in the equation (which might occur if the actual value is greater than the predicted value) we use the square of the error

$$e_j^2 = (f_\theta(x) - y)^2$$

but we have an m number of training examples so we take the sum and average the output.
so the equation becomes:

$$\frac{1}{m}\sum_{i=1}^{m} (f_\theta(x^i) - y^i)^2$$

which means that:
- Linear regression finds parameters $\theta$ that minimize the cost function $j(\theta)$, which represents the average squared errors over the training set.
so the **cost** function becomes:

$$j(\theta) = \frac{1}{m}\sum_{i=1}^{m} (f_\theta(x^i) - y^i)^2$$

- **the half in the equation is added for simplicity.** when differentiating the equation to get the final minimized equation the ($\frac{1}{2}$) makes the math a bit simpler. minimizing the formula with or without the ($\frac{1}{2}$) gives the same results

---
### applying gradient descent in linear regression:
to minimize the cost we already calculated we use the gradient descent (for in depth explanation look into the gradient descent documentation)
we start with the [[Gradient descent]] formula:

$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}j(\theta)$$

#### derivation:
by applying the linear regression formula:

$$\frac{\partial}{\partial\theta_j}j(\theta) =
\frac{\partial}{\partial\theta_j}\sum_{i=1}^{n}\frac{1}{2m}(f_\theta(x)-y)^2$$

##### derivative of $j(\theta)$ with respect to $\theta_0$
step 1: 

$$\frac{\partial}{\partial\theta_0}j(\theta) =
\frac{\partial}{\partial\theta_0}\sum_{i=1}^{n}\frac{1}{2m}(\theta_0+\theta_1x_1-y)^2$$

Applying the chain rule, the derivative of $(u)^2$ with respect to $u$ is $2u$.
step 2:

$$\frac{\partial}{\partial\theta_0}j(\theta) =
\sum_{i=1}^{n}2\frac{1}{2m}(\theta_0+\theta_1x_1-y)$$

step 3:

$$\frac{\partial}{\partial\theta_0}j(\theta) =
\frac{1}{m}\sum_{i=1}^{n}(\theta_0+\theta_1x_1-y)$$

reminder:

$$f(x) = \theta_0+\theta_1x_1$$

now substituting into the original equation for all samples.
so the update formula becomes:

$$\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^{n}(f(x)-y)$$

##### in code:
```
db = (1 / n_samples) * np.sum(y_predicted - y)
bias = bias - learning_rate * db
```
---

derivative of $j(\theta)$ with respect to $\theta_1$
step 1: 

$$\frac{\partial}{\partial\theta_1}j(\theta) =
\frac{\partial}{\partial\theta_1}\sum_{i=1}^{m}\frac{1}{2m}(\theta_0+\theta_1x_1-y)^2$$

Applying the chain rule, the derivative of $(u)^2$ with respect to $u$ is $2u$.
step 2:

$$\frac{\partial}{\partial\theta_1}j(\theta) =
\sum_{i=1}^{m}2\frac{1}{2m}(\theta_0+\theta_1x_1-y)(x)$$

step 3:

$$\frac{\partial}{\partial\theta_1}j(\theta) =
\frac{1}{m}\sum_{i=1}^{m}(\theta_0+\theta_1x_1-y)(x)$$

reminder:

$$f(x) = \theta_0+\theta_1x_1$$

now substituting into the original equation for all samples.
so the update formula becomes:

$$\theta_1 := \theta_1 - \alpha\frac{1}{m}\sum_{i=1}^{m}(f(x)-y)(x)$$

##### in code:
```
dw = (1 / n_samples) * np.dot(x, (y_predicted - y))
weight = weight - learning_rate * dw
```


for full project implementing the linear regression visit the code implementation folder