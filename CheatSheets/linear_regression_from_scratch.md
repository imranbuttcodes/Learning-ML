# 📘 Linear Regression from Scratch (Gradient Descent)

This project explains and implements **Linear Regression using Gradient Descent from scratch** without using sklearn (for learning purposes).

---

# 🚀 Overview

Linear Regression is a supervised machine learning algorithm used to predict continuous values by fitting a straight line to the data.

---

# 🧠 Model Equation

y = mx + b

Where:
- x → input feature
- y → predicted output
- m → slope (weight)
- b → bias (intercept)

---

# 🎯 Goal

Find the best values of `m` and `b` such that:

Predicted values ≈ Actual values

---

# 📉 Loss Function (Mean Squared Error)

MSE = (1/n) * Σ (y - (mx + b))²

Purpose:
- Measures how bad predictions are
- Lower value = better model

---

# ⚙️ Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the loss function by updating parameters step by step.

---

# 🔄 Update Rules

dm = (-2/n) * Σ(x * (y - y_pred))
db = (-2/n) * Σ(y - y_pred)

Parameter updates:

m = m - learning_rate * dm  
b = b - learning_rate * db  

---

# 📌 Key Concepts

- y_pred = m*x + b
- error = y - y_pred
- gradients tell direction of improvement
- learning rate controls step size

---

# 🧪 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# dataset
x = np.array([1,2,3,4])
y = np.array([50,55,65,70])

# initialize parameters
m = 0
b = 0
learning_rate = 0.01
epochs = 100
n = len(x)

# training loop
for i in range(epochs):

    # prediction
    y_pred = m*x + b

    # gradients
    dm = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    # update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Final m:", m)
print("Final b:", b)