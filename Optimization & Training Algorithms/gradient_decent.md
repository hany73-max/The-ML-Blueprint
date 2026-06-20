Gradient descent is an **optimization algorithm** used to minimize a cost function $J(\theta)$.  
It works by **iteratively adjusting** the parameters $\theta$ in the direction that **reduces the cost the fastest** till it reaches convergence.

---
## **Intuition**

for easier understanding of gradient descent we will have to visualize what we are trying to minimize.
lets look at the 2d cost function.


![](../Optimization%20&%20Training%20Algorithms/assets/Screenshot-from-2021-03-03-17-20-40.webp)


minimizing this function meaning we want to get to the lowest point in the function
we will do that by taking the gradient of the function

**gradient reminder:**
- **Gradient direction:** The direction of the greatest (steepest) rate of increase of the function.
- **Gradient magnitude:** The rate at which the function increases in that steepest direction.

so simply the gradient will decide the direction and the rate we need to take to reach the minima by going the opposite way

$$- \frac{\partial}{\partial \theta}j(\theta)$$

next we will multiply the gradient by a rate ($\alpha$) to take smaller steps 
then update the weight ($\theta$) by **subtracting** the output amount from the original weight ($\theta$)

$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}j(\theta)$$

The operator `:=` means **assignment** — the new value of $\theta_j$​ replaces the old one.
##### workflow
- starts from a random point on the surface
    
- calculates the **direction of steepest descent** (gradient)
    
- takes a small step downward
    
- repeats until it reaches the minimum
    

---
##### The Challenge of Non-Convex Landscapes
Simple cost functions (like Mean Squared Error in linear regression) are convex and contain only one global minimum. Complex models (like deep neural networks) present non-convex landscapes featuring:
- **Global Minimum:** The absolute lowest point of the cost function.
    
- **Local Minima:** Valleys where the gradient evaluates to zero in all directions, trapping the model in suboptimal states.

the model might rest in a local minima Because the gradients evaluate to zero in all directions, the model gets "trapped". It will output suboptimal predictions. Because the gradients evaluate to zero in all directions, the model gets "trapped".
that is why there are various types of gradient and other methods to kick the model to reach the global minima
##### Local optimum example (3D mesh illustration):

![](../Optimization%20&%20Training%20Algorithms/assets/1_09kq2L23D9XM_9Xtr8gc8Q.png)

- x marks --> they represent the learning steps the gradient takes 
- the lowest point --> represents the global minima 
- other valleys --> represent other local minimas 

---
## Types of Gradient Descent

- **Batch Gradient Descent**
	
	- Uses **all training samples** to compute each update.
		sums all the samples before updating the $\theta_j$ once
		
	- Very stable but can be slow for large datasets.
	
-  **Stochastic Gradient Descent**
	
	- Uses **one random sample** per update.
		
	- Very fast, but noisy and unstable.
	
- **Mini-Batch Gradient Descent**
	
	- Uses **small batches** of samples (e.g., 32, 64, 128).
		
	- Best of both worlds: fast and stable.
	