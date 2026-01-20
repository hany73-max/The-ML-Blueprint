# regression:

- is a method used to predict a continuous value based on one or more variables 
- it basically tries to find the relation or pattern between variables to create the formula to make best predictions

## Types of regression

##### 1. Linear Regression
	Predicts a straight-line relationship.

##### 2. Polynomial Regression
	Predicts a curve by adding powers of X.

##### 3. Logistic Regression
	Predicts probability (used for classification, despite the name).

##### 4. Ridge / Lasso Regression
	Linear regression + regularization to avoid overfitting.

##### 5. Multiple Regression
	More than one input variable.

more on that later

---


## learning algorithm:
### design
in a simple house pricing data:

| size |  bedrooms | price |
|:----:|:---------:|:-----:|
| 2104 |     3     |  400  |
| 1305 |     2     |  245  |


using the size and price we can describe the relation between them using the linear function:


$$f(x) = \theta_0 + \theta_1x_1 $$


later we can add the number of bedrooms to the formula or add as many variables as we want:


$$f(x) = \theta_0 + \theta_1x_1 + \theta_2x_2$$


adding variables to the equation can make it very long or hard to read. so to make the formula a bit compact we use the summation:


$$f(x) = \sum_{i=0}^{n} \theta_i x_i$$

where $x_0$ = 1
which means:

$$
\theta = 
\begin{bmatrix}
\theta_0 \\ \theta_1 \\ \theta_2 
\end{bmatrix} \quad \quad


x = 
\begin{bmatrix}
x_0 \\ x_1 \\ x_2 
\end{bmatrix}
$$

---
### important notations:
- $\theta$ ----> parameters 
		$\theta_0$ → is the shift in the regression line (**bias**)
		$\theta_n$ → are the ones to define the slope (**weight**)
	
- m ----> number of training examples 
		(number of rows in the data)
	
- $x$ ----> inputs (features)                                            , $n$ ----> number of features
		features are the data we input for the model (values in cells of the data)
	
- $y$ ----> output 
		(prediction or target)
	
- $(x , y)$ ----> training example
		data which consists of features and the prediction so the model can use it to learn the pattern
	
- (xᶦ, yᶦ) ----> the i-th training example
		xᶦ → the features of example i  
		yᶦ → the target for example i
	

---





# References:

- Stanford CS229: Machine Learning - Linear Regression and Gradient Descent | Lecture 2 (Autumn 2018)                   (youtube)