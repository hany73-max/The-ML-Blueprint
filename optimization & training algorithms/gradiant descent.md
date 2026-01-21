Gradient descent is an **optimization algorithm** used to minimize a cost function $J(\theta)$.  
It works by **iteratively adjusting** the parameters $\theta$ in the direction that **reduces the cost the fastest** till it reaches convergence.

---
## **Intuition**

Imagine the cost function as a 3D surface.  
Gradient descent:

- starts from a random point on the surface
    
- calculates the **direction of steepest descent**
    
- takes a small step downward
    
- repeats until it reaches the minimum
    

##### Local optimum example (3D mesh illustration):

![](../material/images/1_09kq2L23D9XM_9Xtr8gc8Q.png)

In practice, with convex functions (like in linear regression), the cost surface has **only one global minimum**.

---

## **Mathematics (General Form)**

Gradient descent updates each parameter using:

​$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}j(\theta)$$

Where:

- $\theta_j$ → the $j$-th parameter
    
- $\alpha$ → learning rate (small value like 0.01)
    
- $\frac{\partial J}{\partial \theta_j}$ → partial derivative (slope) of the cost function
    

The operator `:=` means **assignment** — the new value of $\theta_j$​ replaces the old one.

---

## **Gradient Descent Algorithm**

Repeat until convergence:

1. Compute the gradient$$\frac{\partial J}{\partial \theta_j}$$
2. Update all parameters simultaneously:
$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}j(\theta)$$
3. Check if the cost $J(\theta)$ is still decreasing.

If steps become tiny → the algorithm is converging.

---

## **Short Example: Linear Regression** 

(Full derivation is in **Linear Regression** — this is only a quick illustration.)

For linear regression:
$$j(\theta) = \frac{1}{2}\sum_{i=0}^{m} (f_\theta(x^i) - y^i)^2$$
Its gradient is:
$$\frac{\partial}{\partial\theta_j}j(\theta) = 
2\frac{1}{2}(f_\theta(x^i)-y^i).x_j^i)$$
So the update rule becomes:
$$\theta_j := \theta_j - \alpha\sum_{i=0}^{m}(f_\theta(x^i)-y^i).x_j^i$$
This is **only one example** of gradient descent applied to a specific model.

- For teaching purposes, a “bumpy” surface helps visualize how gradient descent moves downhill.
    
- But in linear regression, the cost function is **quadratic**, so there is only **one global optimum**, not multiple local minima.
    

So the real cost surface for linear regression is smooth and bowl-shaped.

---
## Types of Gradient Descent

- **Batch Gradient Descent**
	
	- Uses **all training samples** to compute each update.
		sums all the samples before updating the $\theta_j$ once
		
	- Very stable but can be slow for large datasets.
	
-  **[[Stochastic Gradient Descent]]**
	
	- Uses **one random sample** per update.
		
	- Very fast, but noisy and unstable.
	
- **[[Mini-Batch Gradient Descent]]**
	
	- Uses **small batches** of samples (e.g., 32, 64, 128).
		
	- Best of both worlds: fast and stable.
	

**in fact the example above and used in linear regression explanation is batch Gradient Descent**

---