### example of a linear regression model:
![..](../../material/images/Screenshot%202026-01-20%20210035.png)
---
## mathematical explanation:
in linear regression we want to minimize the sum of the squared difference between the predicted values and the true training data 

$$(f_\theta(x) - y)^2$$

but we have an m number of training examples so we take the sum.
and the equation becomes:

$$\frac{1}{2m}\sum_{i=0}^{m} (f_\theta(x^i) - y^i)^2$$

which means that:
**- linear regression minimizes the sum of the squared error for m number of examples by minimizing the value of the $j(\theta)$ function.**
so the cost function becomes:

$$j(\theta) = \frac{1}{2m}\sum_{i=0}^{m} (f_\theta(x^i) - y^i)^2$$

**error** ----> is the vertical difference between the true training data and the predicted value

**the half in the equation is added for simplicity.**
when differentiating the equation to get the final minimized equation the ($\frac{1}{2}$) makes the math a bit simpler.
minimizing the formula with or without the ($\frac{1}{2}$) gives the same results

---
## applying gradient descent in linear regression:
we start with the [[Gradient descent]] formula:

$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}j(\theta)$$

### derivation:
by applying the linear regression formula:

$$\frac{\partial}{\partial\theta_j}j(\theta) =
\frac{\partial}{\partial\theta_j}\frac{1}{2m}(f_\theta(x)-y)^2$$

for a single training sample so the sum $\sum$ is removed
after derivation the equation becomes: 

$$\frac{\partial}{\partial\theta_j}j(\theta) = 
2\frac{1}{2m}(f_\theta(x)-y).\frac{\partial}{\partial\theta_j}((f_\theta(x)-y))$$

expanding the $f_\theta(x)$ to it's definition explained in the regression part:

$$\frac{\partial}{\partial\theta_j}j(\theta) = 
2\frac{1}{2m}(f_\theta(x)-y).\frac{\partial}{\partial\theta_j}((x_0\theta_0+x_1\theta_1, ..., x_j\theta_j)-y)$$

reminder:

$$\frac{\partial}{\partial\theta_j}(x_j\theta_j-y) = x_j$$

so the final gradient for a single training sample:

$$\frac{\partial}{\partial\theta_j}j(\theta) = 
2\frac{1}{2m}(f_\theta(x)-y).x_j)$$

now substituting into the original equation for all samples.
so the update formula becomes:

$$\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=0}^{m}(f_\theta(x^i)-y^i).x_j^i$$

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

- Stanford CS229: Machine Learning - Linear Regression and Gradient Descent | Lecture 2 (Autumn 2018)                   (youtube)