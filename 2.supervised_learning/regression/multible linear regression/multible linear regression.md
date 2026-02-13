Multiple linear regression (MLR) is a statistical technique that uses several explanatory variables to predict the outcome of a response variable by creating a plane of best fit to the data.

simply a linear regression model but with more variables.

### Example: multiple Linear Regression Model

![](../multible%20linear%20regression/assets/Screenshot%202026-01-30%20023252.png)

---
## mathematical explanation:

**multiple linear regression** is very close to simple **linear regression** as it also minimizes the sum of the squared residuals.

**residual** ----> is the vertical difference between the true training data and the predicted value.

$$(f_\theta(x) - y)^2$$

but unlike linear regression here we have more than one feature so the linear function becomes:

$$f(x) = \theta_0 + \theta_1x_1 + ... + \theta_jx_j$$

as we have $m$ number of samples.
the equation becomes:

$$\frac{1}{2m}\sum_{i=1}^{m} (f_\theta(x^i) - y^i)^2$$

lastly we can say that the cost function $j(\theta)$ becomes:

$$j(\theta) = \frac{1}{2m}\sum_{i=1}^{m} (f_\theta(x^i) - y^i)^2$$

#### remember:
- **the half in the equation is added for simplicity.** when differentiating the equation to get the final minimized equation the ($\frac{1}{2}$) makes the math a bit simpler. minimizing the formula with or without the ($\frac{1}{2}$) gives the same results

- The difference is squared to eliminate any negative difference which might occur if the actual value is greater than the predicted value.

in this example we will stick with only 2 independent features and the target (the dependent).

$$f(x) = \theta_0 + \theta_1x_1 + \theta_2x_2$$

---

### finding the optimum:
in simple linear regression we directly used the gradient descent method but for multiple linear regression there is more than one approach 

- **The Normal Equation** is a method of finding the optimum beta(𝛽) without iteration. and requires the matrix $X^TX$ to be _invertible_ (though pseudo-inverse solves this).

- **the gradient descent** which uses iterations to find the optimum and requires **Feature Scaling** (normalization) to converge efficiently. (This is a crucial tip for MLR learners!).

here is a comparison between the 2 methods


| gradient descent                | normal equation                   |
| ------------------------------- | --------------------------------- |
| need to choose a learning rate  | no need to choose a learning rate |
| needs many iterations           | no need for iteration             |
| works well with large n-samples | slow with large data-sets         |

in this article we will focus on the gradient descent method as I found it to be easier and service the purpose of this blueprint. 

if u want to check the normal equation u can visit these sources:
- [Multiple Linear Regression from scratch using only numpy by  (debidutta dash)](https://medium.com/analytics-vidhya/multiple-linear-regression-from-scratch-using-only-numpy-98fc010a1926)
- [Derivation of the Normal Equation for linear regression](https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/ "Permalink to Derivation of the Normal Equation for linear regression")

---

#### extracting a general formula for updating $\theta$
starting with the gradient descent formula:

$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}j(\theta)$$

substituting the cost function into the gradient descent formula:

$$\theta_j := \theta_j-\alpha \frac{1}{2m}
\frac{\partial}{\partial\theta_j}\sum_{i=1}^{m} (f_\theta(x^i) - y^i)^2$$

Applying the chain rule, the derivative of $(u)^2$ with respect to $u$ is $2u$.

$$\theta_j := \theta_j-\alpha \frac{1}{m}
\sum_{i=1}^{m} (f_\theta(x^{(i)}) - y^{(i)}).x_j^{(i)}$$

---

#### vectorization
In professional machine learning (like in TensorFlow or PyTorch), we almost never loop through features individually ($\theta_1, \theta_2, \dots$). Instead, we bundle all the features and weights into **Matrices** and **Vectors** to calculate everything in one single shot.

instead of walking thro every feature by hand like we did for simple linear regression, we can this **vectorization** method.
first we need to change how we look at the data.

**The Weights ($\theta$):** Instead of separate variables (`bias`, `w1`, `w2`), we stack them into a single column vector. 

$$\theta = \begin{bmatrix}
\theta_0 \\ \theta_1 \\ \vdots \\ \theta_i 
\end{bmatrix}$$

**The Features ($X$):** We stack all our training samples into a big matrix.

$$
X =
\begin{bmatrix}
1 & x_1^{(1)} & \cdots & x_i^{(1)} \\
1 & x_1^{(2)} & \cdots & x_i^{(2)} \\
\vdots & \vdots &   & \vdots\\
1 & x_1^{(m)} & \cdots & x_i^{(m)}
\end{bmatrix}
$$

$m$ ----> n_samples

Now, instead of writing $\theta_0 + \theta_1x_1 + \theta_2x_2$, we just perform a matrix multiplication (Dot Product):

$$f(x) = X \cdot \theta$$

of course the actual target will also get stacked into a matrix:

$$y = \begin{bmatrix} 
y^1 \\ y^2 \\ \vdots \\ y^m
\end{bmatrix}$$

we stated the error as $(y_{predicted} - y)$ or $(f(x) - y)$ in the linear regression article.

here it will be the same but using vectors.

$$\text{Error Vector } (E) = X\theta - y = \begin{bmatrix}
\text{pred}^{(1)} - y^{(1)} \\ 
\text{pred}^{(2)} - y^{(2)} \\ 
\vdots \\ 
\text{pred}^{(m)} - y^{(m)}
\end{bmatrix}$$

Look at the derivative formula again: **Sum of (Error $\times$ Feature)**. We need to multiply the **Error Column** by the **Feature Column** and sum the results.

In linear algebra, to multiply "Feature $\times$ Error" and sum them up, we use the **Dot Product**.

$$X \cdot E$$

Let’s look at the shapes of your data:
- **Matrix $X$:** $(m \times n)$ — ($m$ rows of samples, $n$ columns of features).
- **Vector $E$ (Error):** $(m \times 1)$ — (One error value for every sample).

the inner numbers (**$n$** and **$m$**) do not match! You cannot perform this multiplication unless the number of features equals the number of samples, which is rarely the case.

**The Solution: Transpose X ($X^T$)** We flip $X$ on its side. Now, the **rows** represent features, and the **columns** represent samples.

So the final complete formula is:

$$\theta := \theta - \alpha \cdot \frac{1}{m} \cdot \underbrace{X^T \cdot (X\theta - y)}_{\text{The Gradient}}$$

```
for i in range(iterations):
    # 1. Calculate predictions for all samples
    predictions = np.dot(X_vectorized, theta)
    
    # 2. Calculate errors
    error = predictions - y
    
    # 3. Calculate gradient
    gradient = (1/m) * np.dot(X_vectorized.T, error)
    
    # 4. Update theta
    theta = theta - (learning_rate * gradient)

# Extract final values
bias_final = theta[0]
weights_final = theta[1:]
```
for full code implementation projects u can check the code implementation folder
