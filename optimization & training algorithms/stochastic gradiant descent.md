SGD uses the general update formula for gradient descent:
$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}j(\theta)$$
Instead of computing the gradient using **all training examples** at once (like Batch GD), SGD updates the parameters **one training example at a time**:

- take a single example $(x^i,y^i)$
    
- compute its error
    
- update θ immediately
    
- move to the next example
    
- repeat for all $m$ examples
    

This makes the updates **noisy and jumpy** because each example pulls θ in a slightly different direction.  
So it doesn’t always follow the “straight path” downhill.

### Advantages:

- **More practical for large datasets** (doesn’t need full dataset in memory)
    
- **Faster updates** (1 sample → quick gradient)
    
- **Can escape local minima** because of noise
    
- Good for **online learning** (streaming data)
    
### Disadvantages:

- **Noisy / Random** updates
    
- May **oscillate** around the minimum
    
- Might **never converge exactly**, but usually gets close enough

### SGD graph:
![](../material/images/Pasted%20image%2020251206062232.png)

---