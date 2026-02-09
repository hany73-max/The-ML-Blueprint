Linear Regression is a statistical tool that produces a line of best fit for a given dataset analytically. To produce the regression line manually, one needs to perform operations such as mean-squared error and optimizing the cost function; both are explained in detail later.
### Example: Linear Regression Model
![](../linear%20regression/assets/Screenshot%202026-01-20%20210035.png)
## mathematical explanation:
in linear regression we want to minimize the sum of the squared difference between the predicted values and the true training data 

$$(f_\theta(x) - y)^2$$

but we have an m number of training examples so we take the sum.
and the equation becomes:

$$\frac{1}{2m}\sum_{i=1}^{m} (f_\theta(x^i) - y^i)^2$$

which means that:
- Linear regression finds parameters $\theta$ that minimize the cost function $j(\theta)$, which represents the average squared error over the training set.

so the **cost** function becomes:

$$j(\theta) = \frac{1}{2m}\sum_{i=1}^{m} (f_\theta(x^i) - y^i)^2$$

**residual** ----> is the vertical difference between the true training data and the predicted value

- **the half in the equation is added for simplicity.** when differentiating the equation to get the final minimized equation the ($\frac{1}{2}$) makes the math a bit simpler. minimizing the formula with or without the ($\frac{1}{2}$) gives the same results

- The difference is squared to eliminate any negative difference which might occur if the actual value is greater than the predicted value.

---
## applying gradient descent in linear regression:
we start with the [[Gradient descent]] formula:

$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}j(\theta)$$

### derivation:
by applying the linear regression formula:

$$\frac{\partial}{\partial\theta_j}j(\theta) =
\frac{\partial}{\partial\theta_j}\sum_{i=1}^{m}\frac{1}{2m}(f_\theta(x)-y)^2$$

derivative of $j(\theta)$ with respect to $\theta_0$

step 1: 

$$\frac{\partial}{\partial\theta_0}j(\theta) =
\frac{\partial}{\partial\theta_0}\sum_{i=1}^{m}\frac{1}{2m}(\theta_0+\theta_1x_1-y)^2$$

Applying the chain rule, the derivative of $(u)^2$ with respect to $u$ is $2u$.

step 2:

$$\frac{\partial}{\partial\theta_0}j(\theta) =
\sum_{i=1}^{m}2\frac{1}{2m}(\theta_0+\theta_1x_1-y)$$

step 3:

$$\frac{\partial}{\partial\theta_0}j(\theta) =
\frac{1}{m}\sum_{i=1}^{m}(\theta_0+\theta_1x_1-y)$$

reminder:

$$f(x) = \theta_0+\theta_1x_1$$

##### in code:

```
y_pred = np.dot(x, weight) + bias
```

now substituting into the original equation for all samples.
so the update formula becomes:

$$\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^{m}(f(x)-y)$$

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

for a full project implementing the linear regression visit the code implementation folder

---

## gradient descent graphs in linear regression

- The cost function is **quadratic**, so there is only **one global optimum**
    
- The cost surface is smooth and bowl-shaped
    
### example:
![](../../material/images/2019-12-11-LinReg-fig1.png)

to show the iterations ($\alpha$) a bit better here is a 
### vertical view of the mesh:
![](../../material/images/Pasted%20image%2020251205130347.png)

- Gradient descent takes steps toward the lowest $j(\theta)$
    
- If $\alpha$ is too large → may overshoot
    
- If $\alpha$ is too small → takes many iterations to converge

--- 
# References:

- Stanford CS229: Machine Learning - Linear Regression and Gradient Descent | Lecture 2 (Autumn 2018) for          **(andrew ng)**
- Mathematics Behind Linear Regression paper for         **(rahul ravi)**