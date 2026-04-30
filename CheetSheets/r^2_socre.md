# 🧠 R² (R-Squared) – “How much better is my model than the mean?” 🔥

R² is a key regression metric that answers:  
👉 **“How much better is my model than just predicting the average of y?”**

---

## 🎯 Core idea

- **Dumb baseline model:** always predicts \(\bar{y}\) (mean of target).
- **Your ML model:** predicts \(\hat{y} = mx + b\) using patterns.
- **R² compares** the error of your model vs the error of that dumb baseline.

---

## 📊 Step 1: Define the baseline

Given data:

| x | y  |
|---|----|
| 1 | 50 |
| 2 | 55 |
| 3 | 65 |
| 4 | 70 |

Compute mean:
\[
\bar{y} = \frac{50 + 55 + 65 + 70}{4} = 60
\]
So the dumb model always predicts \(\hat{y} = 60\).

---

## 📉 Step 2: Total error of the baseline (SST)

This is the **Total Sum of Squares (SST)**:
\[
\text{SST} = \sum (y - \bar{y})^2
\]

| y  | \(y - 60\) | \((y - 60)^2\) |
|----|-----------|----------------|
| 50 | -10       | 100            |
| 55 | -5        | 25             |
| 65 | 5         | 25             |
| 70 | 10        | 100            |

\[
\text{SST} = 100 + 25 + 25 + 100 = 250
\]

👉 **SST = how spread out the data is from the mean.**

---

## 📉 Step 3: Error of your model (SSE)

This is the **Sum of Squared Errors (SSE)**:
\[
\text{SSE} = \sum (y - \hat{y})^2
\]

Suppose your model predicts:
\[
\hat{y} = [52, 56, 64, 68]
\]

Then:
\[
\text{SSE} \approx (50 - 52)^2 + (55 - 56)^2 + (65 - 64)^2 + (70 - 68)^2
\]
\[
\text{SSE} = 4 + 1 + 1 + 4 = 10
\]

👉 **SSE = how wrong your model is after fitting the line.**

---

## 🧠 Step 4: R² formula

\[
R^2 = 1 - \frac{\text{SSE}}{\text{SST}}
\]

Plug in the numbers:
\[
R^2 = 1 - \frac{10}{250} = 1 - 0.04 = 0.96
\]

👉 **Your model explains 96% of the variation in y.**

---

## 🎯 Interpreting R² values

| R² value | Meaning |
|----------|---------|
| 1.0      | Perfect model (SSE = 0) |
| 0.8–1.0  | Very good model |
| 0.5–0.8  | Moderate explanatory power |
| ≈ 0.0    | No better than predicting the mean |
| < 0      | Worse than the mean |

---

## 🐍 Python example (sklearn)

```python
from sklearn.metrics import r2_score

y_true = 
y_pred =   # your model predictions

r2 = r2_score(y_true, y_pred)
print("R² =", r2)
# Output: R² ≈ 0.96
```

---

## 🧠 Intuition you can remember

- **R² = 1** → “Your model explains all the variation.”
- **R² = 0** → “Your model is useless; just use the mean.”
- **R² negative** → “Your model is worse than the mean.”

**One‑line memory trick:**  
👉 **“R² = how much better is my model than just using the mean?”**

---

## 🚀 Next step (if you want)

You can now:
- Calculate R² directly in sklearn.
- Understand how it appears in LinearRegression `.score()`.  
- Move to Logistic Regression and classification metrics.

---

