# Linear Regression & Multiple Linear Regression — Complete Reference
> ML Algorithms Series | Imran's Study Notes

---

## What is Linear Regression?

A supervised ML algorithm that models the relationship between input features and a **continuous numerical output** by fitting the best possible line (or hyperplane) through the data.

---

## Equations

### Simple Linear Regression (1 feature)
```
y = β₀ + β₁x

β₀ = intercept (y value when x = 0)
β₁ = slope (how much y changes per unit increase in x)
```

### Multiple Linear Regression (n features)
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Each βᵢ = how much y changes when xᵢ increases by 1,
           holding all other features constant
```

---

## How It Works — OLS

Finds β values that minimize **RSS (Residual Sum of Squares)**:

```
Residual  = Actual y - Predicted y        (error per point)
RSS       = Σ (yᵢ - ŷᵢ)²                (sum of squared errors)
```

The algorithm finds the β values that make RSS as small as possible → **Ordinary Least Squares (OLS)**.

---

## Full Code

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = df[['experience', 'age']]
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("Intercept:   ", model.intercept_)
print("Coefficients:", model.coef_)
print("MAE: ",  mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:  ", r2_score(y_test, y_pred))
```

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **R²** | 1 - RSS/TSS | % of variance explained. 0.85 = model explains 85% |
| **Adjusted R²** | Penalizes extra features | Use for MLR to compare models fairly |
| **MAE** | mean(\|actual - predicted\|) | Avg error in original units, easy to interpret |
| **RMSE** | √mean((actual - predicted)²) | Like MAE but penalizes large errors more |

```python
# Adjusted R²
n   = len(y_train)
p   = X_train.shape[1]
r2  = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

> R² = 1.0 → suspicious (likely overfitting). R² < 0 → model is worse than predicting the mean.

---

## Key Assumptions

| # | Assumption | How to Check |
|---|---|---|
| 1 | **Linearity** — X and y have a linear relationship | Scatter plot each feature vs target |
| 2 | **No Multicollinearity** — features not correlated with each other | VIF score (>5 = concern, >10 = serious) |
| 3 | **Homoscedasticity** — residuals have constant variance | Plot residuals vs predicted values |
| 4 | **Normality of residuals** — errors are normally distributed | Q-Q plot |
| 5 | **No Autocorrelation** — residuals are independent | Durbin-Watson test (~2.0 is good) |

```python
# Check multicollinearity (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['feature'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Check residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0)  # should look like a random cloud, not a funnel

# Durbin-Watson
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(residuals))  # ~2.0 = no autocorrelation
```

---

## Where to Use

| ✅ Good Use Cases | ❌ Bad Use Cases |
|---|---|
| Salary prediction | Classification problems |
| House price prediction | Image recognition |
| Sales forecasting | Non-linear relationships |
| Risk scoring (baseline) | Text data |
| Any roughly linear relationship | Time series with trends/seasonality |

---

## Pros and Cons

### ✅ Pros
| Advantage | Why It Matters |
|---|---|
| Simple and interpretable | Coefficients explain exactly why each prediction was made |
| Fast to train | OLS is computationally cheap even on large datasets |
| Great baseline model | Always build LR first before complex models |
| Feature importance built-in | Coefficients directly show feature impact |
| No hyperparameter tuning (basic) | Just train and go |

### ❌ Cons
| Disadvantage | When It Hurts |
|---|---|
| Assumes linear relationship | Real-world data is rarely perfectly linear |
| Sensitive to outliers | One extreme point drags the line significantly |
| Multicollinearity breaks it | Correlated features make coefficients unreliable |
| Can't handle non-linearity natively | Needs feature engineering or a different model |
| Underfits complex data | High bias — misses complex patterns |
| No built-in feature selection | Use Lasso for that |

---

## SLR vs MLR

| | Simple LR | Multiple LR |
|---|---|---|
| Features | 1 | 2 or more |
| Equation | y = β₀ + β₁x | y = β₀ + β₁x₁ + β₂x₂ ... |
| Visualization | 2D line | Hyperplane (can't visualize easily) |
| Multicollinearity risk | None | Yes — always check VIF |
| Use Adjusted R²? | No | Yes |

---

## Regularized Versions

Use when: too many features, multicollinearity, or overfitting.

| Model | Penalty | Effect | Use When |
|---|---|---|---|
| **Ridge (L2)** | Sum of β² | Shrinks all coefficients | Multicollinearity present |
| **Lasso (L1)** | Sum of \|β\| | Shrinks some to exactly 0 | Feature selection needed |
| **ElasticNet** | L1 + L2 mix | Combination of both | Both problems present |

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge   = Ridge(alpha=1.0)
lasso   = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

> ⚠️ Always apply StandardScaler before regularized regression.

---

## Common Mistakes

| Mistake | Why It's Wrong | Fix |
|---|---|---|
| Using LR for classification | Predicts values outside [0,1] | Use Logistic Regression |
| Not checking multicollinearity | Unstable, unreliable coefficients | Check VIF, drop/combine correlated features |
| Not scaling before regularization | Penalty unfairly affects small-scale features | Always StandardScaler first |
| Trusting R² alone for MLR | R² always goes up with more features | Use Adjusted R² |
| Ignoring residual plots | Assumption violations go undetected | Always plot residuals |

---

## Quick Reference

```
Linear Regression is right when:
  ✅ Target is continuous (price, salary, temperature)
  ✅ Relationship looks roughly linear
  ✅ You need interpretability
  ✅ You want a fast baseline

Switch to something else when:
  ❌ Target is a category → Logistic Regression / Classification
  ❌ Relationship is curved → Polynomial Regression / Tree models
  ❌ Too many correlated features → Ridge / Lasso
  ❌ Complex patterns → Random Forest / XGBoost / Neural Networks
```

---

