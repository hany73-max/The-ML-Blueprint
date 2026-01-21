Mini-batch gradient descent is a middle ground between **Batch GD** and **SGD**.  
It uses the same update formula:
$$\theta_j := \theta_j-\alpha \frac{\partial}{\partial\theta_j}j(\theta)$$
Instead of using:

- **1 example** (SGD), or
    
- **all examples** (Batch GD),
	

mini-batch splits the training data into **small batches**, typically 16, 32, 64, or 128 samples.
Then for each mini-batch:

- compute the gradient for that small batch
    
- update θ
    
- move to the next mini-batch
    
- repeat until the whole dataset is processed
    

This gives updates that are **more stable than SGD** and **much faster than Batch GD**.

### Advantages

- **Faster** than batch GD
    
- **More stable** than SGD (less noisy)
    
- Works extremely well with GPUs
    
- Most commonly used in deep learning
    
- Good balance between speed and accuracy
    

### Disadvantages

- Still has some noise
    
- Requires choosing a **batch size**
    
- Not as stable as full batch GD

### mini-batch GD graph:
![](../material/images/Pasted%20image%2020251206063142.png)

---