# 🏛️ Architecture of Intelligence  
## A First-Principles Approach to Machine Learning

> **Purpose:** To explain machine learning models from their underlying mathematical principles, reducing reliance on abstraction and revealing how modern algorithms actually work.

---

## 🎯 Conceptual Focus

Modern machine learning frameworks make it possible to build powerful systems quickly, but often at the cost of obscuring the mechanics that govern learning and generalization.

This repository exists to **make those mechanics explicit**.

The goal is not to replace high-level tools, but to **understand what they implement** — by studying models through their conceptual motivation, mathematical formulation, and algorithmic structure.

---

## 🧠 Methodology

Each topic in this repository follows a consistent first-principles workflow:

1. **Conceptual Theory**  
   - What problem does the model solve?  
   - What assumptions does it make about data?

2. **Mathematical Formulation**  
   - Objective functions and constraints  
   - Loss definitions and optimization goals  
   - Derivations that connect intuition to equations

3. **Algorithmic Interpretation**  
   - How the mathematics translates into an iterative or closed-form procedure  
   - Update rules and convergence behavior

4. **Empirical Insight (When Applicable)**  
   - Visualization of learning dynamics  
   - Geometric or statistical interpretation of results  

Not every model is implemented in full.  
For more complex algorithms, **mathematical understanding takes priority over code completeness**.

---

## 🗺️ Learning Roadmap

The repository is organized as a progressive reference, starting from foundational models and moving toward more expressive learning systems.

### 📍 Phase I: Supervised Learning & Optimization
- **Linear Models**  
  Least Squares, objective geometry, and the Normal Equation
- **Optimization Fundamentals**  
  Gradient Descent, Stochastic Methods, and convergence intuition
- **Generalization Theory**  
  Bias–Variance Tradeoff and the role of Regularization

### 📍 Phase II: Discriminative Models *(In Development)*
- **Logistic Models**  
  From regression to classification and decision boundaries
- **Information-Based Learning**  
  Entropy, Information Gain, and Decision Trees

---

## 🛠️ Core Principles

- **Math-First Explanations**  
  Every model is defined by its mathematics before any algorithmic form is introduced.

- **Minimal Abstraction**  
  When implementations are included, they rely on basic numerical operations to preserve transparency.

- **Explicit Reasoning**  
  No hidden steps, no unexplained shortcuts.

- **Visual Intuition**  
  Plots and diagrams are used to connect equations to behavior.

---

## 📂 Repository Structure *(Evolving)*

```text
├── supervised_learning/
│   ├── regression/             # Mathematical derivations and conceptual notes
│   ├── optimization/           # Learning mechanics and update rules
│   ├── implementations/        # Selected models implemented from first principles
│   └── evaluation/             # Generalization, metrics, and validation theory
└── material/
    └── images/                 # Visualizations and geometric interpretations
