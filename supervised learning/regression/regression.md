# regression:

- is a method used to predict a continuous value based on one or more variables 
- it basically tries to find the relation or pattern between variables to create the formula to make best predictions

## Types of regression

##### [[Linear Regression]]
	Predicts a straight-line relationship.

##### Multiple Regression
	Same as linear regression but more than one input variable.

##### Polynomial Regression
	Predicts a curve by adding powers of X.

##### Ridge / Lasso Regression
	Linear regression + regularization to avoid overfitting.

etc...
more on that later

---


## learning algorithm:
### design
in a simple house pricing data:

| size | price |
|:----:|:-----:|
| 2104 |  400  |
| 1305 |  245  |


using the size and price we can describe the relation between them using the linear function:


$$f(x) = \theta_0 + \theta_1x_1 $$


later we can add the number of bedrooms to the formula or add as many variables as we want:


$$f(x) = \theta_0 + \theta_1x_1 + \theta_2x_2$$


adding variables to the equation can make it very long or hard to read. so to make the formula a bit compact we use the summation:


$$f(x) = \sum_{i=0}^{m} \theta_i x_i$$

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
	
- $n$ ----> number of training examples 

		(number of rows in the data)
	
- $x$ ----> inputs (features)                                            , $m$ ----> number of features

		features are the data we input for the model (the columns)
	
- $y$ ----> output 

		(prediction or target)
	
- $(x , y)$ ----> training example

		data which consists of features and the prediction so the model can use it to learn the pattern
	
- (xᶦ, yᶦ) ----> the i-th training example

		xᶦ → the features of example i  

		yᶦ → the target for example i

---