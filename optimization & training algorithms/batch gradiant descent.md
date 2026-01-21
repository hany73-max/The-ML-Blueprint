batch GD uses the general update formula for gradient descent:
$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}j(\theta)$$
Batch GD calculates the gradient using **the entire training dataset at once**:

- take all $m$ training examples
    
- compute the total cost
    
- compute the total gradient
    
- then update **θ once per epoch**
    

This means every update is based on the **most accurate and stable direction downhill**.

### Advantages

- **Stable and smooth** convergence
    
- Moves in the “true” steepest direction
    
- Works well for **small datasets**
    
- Usually converges to the minimum no matter what
    

### Disadvantages

- **Very slow** on large datasets
    
- Needs the entire dataset in memory
    
- One update = very expensive
    
- Not suitable for real-time / online learning

### batch GD graph:
![](../material/images/Pasted%20image%2020251206062603.png)

---