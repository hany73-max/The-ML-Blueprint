# **AI vs ML**
## **Artificial Intelligence (AI)**
is a field of computer science that develops systems capable of performing tasks typically requiring human intelligence, such as reasoning, learning, problem-solving, and perception.
simply a program that can use information to make decisions or predictions without active human involvement.
## **machine learning (ML)**
**Machine learning**: is the subset of artificial intelligence (AI) focused on algorithms that can “learn” the patterns of training data and, subsequently, make accurate _inferences_ about new data. This pattern recognition ability enables machine learning models to make decisions or predictions without explicit, hard-coded instructions.

**the fundamental goal of machine learning**: (a trained model) is to apply patterns it learned from training data to infer the correct output for a real-world task

---
# **Types of machine learning**
---
## **1. Supervised machine learning**

a type of machine learning where the model is trained on a labelled dataset

**The objective of any supervised machine learning algorithm** is to optimize model parameters in a way that minimizes the output of a **loss function** that measures the divergence (“loss”) between a model’s predicted output for each input and the ground truth output for each of those inputs.

### **Supervised ML algorithms**

#### **Classification:**
 entails discrete predictions, such as the specific category a data point belongs to. Classification tasks are either _binary—_”yes” or “no”_, “_approve” or “reject_,” “_spam” or “not spam”_—_or _multi-class._ Some classification algorithms are only suited to binary classification, but any multi-class classifier can perform binary classification.
 
 **Common classification algorithms:**
 
 - ##### Naive bayes: 
	 classifiers operate on the logic of Bayes’ Theorem, which is essentially a mathematical formulation of the idea that information from later events. In other words, the model learns the relative importance of a given input variable based on how strongly it correlates with specific outcomes
	
- ##### Logistic regression:
	adapts the linear regression algorithm to solve binary classification (0 & 1) problems by  feeding the weighted sum of input features into a sigmoid function
	
- ##### K-nearest neighbor (KNN): 
	algorithms classify data points based on their proximity in the vector embedding space to   other. The _k_ refers to how many neighboring data points are taken into consideration
	
- #####  Support vector machines (SVMs): 
	An SVM algorithm’s goal is not to directly learn how to categorize data points: instead, its  goal is to learn the optimal _decision boundary_ to separate two categories of labeled data points in order to then classify new data points based on which side of the boundary they fall. The SVM algorithm defines this boundary as the hyperplane that maximizes the margin (or gap) between data points of opposite classes
	

#### **Regression:**
is used to predict _continuous_ values, such as quantities, prices, duration or temperature.

**Common regression algorithms:**
- ##### Linear regression:
	Linear Regression is a statistical tool that produces a line of best fit for a given dataset analytically. To produce the regression line manually, one needs to perform operations such as mean-squared error and optimizing the cost function
	
- ##### Decision tree algorithms:
	each final output predictions through a branching sequence of if-then-else decisions that can be visualized as a tree-like diagram. They can be utilized for both regression and classification.
	
- ##### State space models:
	model dynamic systems and sequential data through two interrelated equations: one, the _state equation,_ describes the internal dynamics (“state”) of a system that aren’t directly observable; the other, the _output equation,_ describes how those internal dynamics relate to observable results—that is, to system outputs
	

---
## **2. Unsupervised learning** 
used to teach models to discover intrinsic patterns, correlations and structure in unlabeled data.

 their tasks don’t entail a known ideal output to measure against and optimize toward. The success of the learning process is governed primarily by manual hyperparameter tuning, rather than through algorithms that optimize the model’s internal parameters.

There are three fundamental subsets of unsupervised learning algorithms
### **Unsupervised ML algorithms**:
#### Clustering algorithms:
 partition unlabelled data points into “clusters,” or groupings, based on their proximity or similarity to one another.

- ##### K-means clustering :
	algorithms partitions data into _k_ clusters in a given data point will be assigned to the cluster whose center (_centroid_) it’s closest to. then Each centroid is then relocated to the position representing the average (_mean_) of all the data points that were just assigned to it. and the cycle is repeated until the centroid is stabilized.
	
- ##### Gaussian mixture models (GMMs):
	A GMM assumes that a dataset is a mixture of multiple Gaussian—that is, classic “normal" or “bell curve”—distributions, and predicts the probability that a given data point belongs to each of those normally distributed clusters.
	
- ##### DBSCAN (Density-Based Spatial Clustering Applications with Noise):
	creates clusters out of data points that are closely packed together. Rather than grouping all data points into clusters, it marks data points located alone in low-density regions as outliers.
	

#### Association algorithms:
Association algorithms identify correlations between variables in large datasets.

- ##### The apriori algorithm:
	a classic association method. The algorithm makes multiple passes over the dataset using a "bottom-up" approach, exploring the frequency of progressively larger combinations of individual items and pruning combinations that appear infrequently.
	
- ##### Dynamic itemset counting (DIC): 
	is a more compute-efficient association method, though it operates through logic similar to that of the classic apriori algorithm. Rather than exploring the entire dataset with each pass, it starts with only a subset of the database and then periodically adds new items, “dynamically” expanding its focus.
	

#### Dimensionality reduction algorithms:
they’re designed to learn a mapping of high-dimensional data points to a space where they can be accurately described using fewer features: in other words, to _reduce_ the number of _dimensions_ needed to represent data effectively.

- ##### Principal component analysis (PCA):
	simplifies complex datasets by summarizing the data’s original variables—many of which are often correlated with one another, and thus somewhat redundant—as a smaller subset of uncorrelated variables, each of which is a linear combination of original variables.
	
- ##### T-distributed Stochastic Neighbor Embedding (t-SNE):
	A non-linear dimensionality reduction algorithm commonly used for data visualization purposes. It’s used almost exclusively to represent data in either 2 or 3 dimensions.
	
- ##### Autoencoders:
	A type of encoder-decoder neural network architecture trained through what might more commonly be considered a self-supervised learning algorithm.
	The encoder comprises a series of progressively smaller layers, squeezing the data into fewer and fewer dimensions before it reaches the decoder.
	The decoder, which comprises a series of progressively _larger_ layers, is then tasked with reconstructing the original data using this compressed representation, with the objective of minimizing _reconstruction loss_.
	

---
## **3. Reinforcement learning**
algorithms are suited for tasks in which there’s no singular “correct” output (or action), but there are “good” outputs.
Rather than a supervisory signal and explicitly defined tasks, they entail a _reward signal_ that allows models to learn holistically through trial and error. That reward signal can come from a reward function, a separately trained reward model, or a rules-based reward system.

RL algorithms can be _value-based_ or _policy-based._ In policy-based algorithms, a model learns an optimal policy directly. In value-based algorithms, the agent learns a value function that computes a score for how “good” each state is—typically based on the potential reward for actions that can be taken from that state—then chooses actions that lead to higher-value states. Hybrid approaches learn a value function that, in turn, is then used to optimize a policy.
