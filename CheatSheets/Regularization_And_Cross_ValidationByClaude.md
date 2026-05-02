# Cross Validation & Regularization — Complete Reference Guide
> ML Concepts Series | Imran's Study Notes

---

# PART 1 — CROSS VALIDATION

---

## The Problem With a Simple Train-Test Split

You've been doing this:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Seems fine. But here's the issue.

Your test set is just **one random slice** of the data. What if by luck, all the easy
data points landed in the test set? Your R² looks great — but your model is actually bad.
You just got lucky with the split.

```python
# Same model, same data — different random split, completely different score
train_test_split(X, y, random_state=42)  →  R² = 0.91
train_test_split(X, y, random_state=0)   →  R² = 0.76
train_test_split(X, y, random_state=7)   →  R² = 0.83

# Which one is the real performance? Nobody knows.
```

Your evaluation shouldn't depend on luck. **Cross Validation solves this.**

---

## What is Cross Validation?

Instead of splitting once, you split **multiple times** and average the results.
Every data point gets a fair chance to be in the test set.

### How 5-Fold CV Works (Step by Step)

```
Your training data — imagine 10 rows:
[1] [2] [3] [4] [5] [6] [7] [8] [9] [10]

Split into 5 equal folds:
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Fold 1  │ Fold 2  │ Fold 3  │ Fold 4  │ Fold 5  │
│  [1,2]  │  [3,4]  │  [5,6]  │  [7,8]  │ [9,10]  │
└─────────┴─────────┴─────────┴─────────┴─────────┘

Round 1: [TEST][train][train][train][train] → Score: 0.88
Round 2: [train][TEST][train][train][train] → Score: 0.91
Round 3: [train][train][TEST][train][train] → Score: 0.85
Round 4: [train][train][train][TEST][train] → Score: 0.90
Round 5: [train][train][train][train][TEST] → Score: 0.87

Final CV Score = (0.88 + 0.91 + 0.85 + 0.90 + 0.87) / 5 = 0.882
```

Every single data point gets to be in the test set **exactly once**.
No luck. The score is honest and reliable.

---

## Code

### Basic Cross Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

model = LinearRegression()

# cv=5 means 5 folds
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

print("Score per fold:", scores)
# Output: [0.88, 0.91, 0.85, 0.90, 0.87]

print("Mean CV R²:    ", scores.mean())
# Output: 0.882  ← this is your real performance estimate

print("Std deviation: ", scores.std())
# Output: 0.021  ← how stable the model is across different splits
```

**Mean** → real performance estimate
**Std** → model stability. Low std = consistent. High std = model is too sensitive to data.

---

### cross_val_score vs cross_validate

```python
# cross_val_score → only returns test scores (most common)
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# cross_validate → returns both train AND test scores
from sklearn.model_selection import cross_validate

results = cross_validate(model, X, y, cv=5,
                         scoring='r2',
                         return_train_score=True)

print("Train scores:", results['train_score'])
print("Test scores: ", results['test_score'])
```

Getting both scores tells you exactly what's going on:

```
Train R² high + Test R² high  →  Good fit             ✅
Train R² high + Test R² low   →  Overfitting           ❌
Train R² low  + Test R² low   →  Underfitting          ❌
Train R² low  + Test R² high  →  Impossible (red flag) ⚠️
```

---

## How Many Folds?

```
cv=5   → most common, good balance of speed and reliability
cv=10  → more reliable, slower to compute
cv=n   → called Leave One Out CV (LOOCV)
         most reliable, extremely slow on large datasets
         n = total number of training samples
```

**Standard choice: cv=5 for most projects.**

---

## Critical Rule — CV Does NOT Replace Your Test Set

This is the most common mistake beginners make.

```
❌ WRONG workflow:
Full data → CV on everything → report CV score as final result

✅ CORRECT workflow:

Full Data
    ↓
Split into Train (80%) and Test (20%)
Test set is LOCKED AWAY — never touched until the very end
    ↓
Use CV only on Train data to:
    - Compare models
    - Find best hyperparameters
    - Estimate performance
    ↓
Once model is finalized → evaluate ONCE on Test set
    ↓
That final test score = your real, honest result
```

Think of it like this:
- **CV** = practice exams (used to study and tune)
- **Test set** = the final exam (used only once, at the very end)

You don't use the final exam for practice. You don't cheat by looking at it early.

---

## When to Use CV

```
✅ Comparing two different models (which is better?)
✅ Finding the best polynomial degree
✅ Finding the best alpha (Ridge, Lasso)
✅ Checking if a model is stable
✅ Any hyperparameter tuning

❌ NOT as a replacement for final test set evaluation
❌ NOT on your full dataset (only on training portion)
```

---

## Finding Best Hyperparameters — GridSearchCV

GridSearchCV automates the process of trying different values and using CV to find the best one.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge())
])

# Dictionary of values to try
param_grid = {'model__alpha': [0.01, 0.1, 1, 10, 100, 1000]}

# Try all alpha values using 5-fold CV each time
grid_search = GridSearchCV(pipeline, param_grid,
                           cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best alpha:", grid_search.best_params_)
# Output: {'model__alpha': 10}

print("Best CV R²:", grid_search.best_score_)
# Output: 0.9341

# Use the best model for final prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

---

## Summary

| Concept | What It Does |
|---|---|
| `cross_val_score` | Splits training data K times, returns K scores |
| `cross_validate` | Same but also returns train scores |
| `GridSearchCV` | Tries all hyperparameter combos using CV |
| `cv=5` | Standard choice — 5 folds |
| Mean CV score | Your honest performance estimate |
| Std of scores | How stable/consistent your model is |

---
---

# PART 2 — REGULARIZATION

---

## The Problem It Solves — Overfitting

When a model overfits, it memorizes the training data — including noise.

```
Training data (salary by experience):

Experience:  1     2     3     4     5
Salary:    30k   33k   41k   48k   52k

The 33k might just be noise — maybe that person had a bad negotiation.
The 41k might be noise too — bonus year.

Overfit model: "I will memorize ALL points exactly, including noise."
Good model:    "I'll learn the general trend, ignore the noise."
```

**How does overfitting happen technically?**

The model makes its β coefficients **extremely large** to hit every training point perfectly.

```
Overfit model:
Salary = 5000 + 80000×x₁ - 60000×x₂ + 30000×x₃ ...
                ↑ these giant β values → wild, extreme predictions on new data
```

**Regularization adds a penalty for large β values** — forcing the model to keep
coefficients small, the curve smooth, and predictions sensible on new data.

---

## The Cost Function — How Penalty Works

```
Normal Linear Regression minimizes only RSS:
    Cost = Σ(actual - predicted)²

Ridge adds a penalty term:
    Cost = Σ(actual - predicted)² + α × Σ(β²)
                                    ↑
                              penalty for large β

Lasso adds a different penalty:
    Cost = Σ(actual - predicted)² + α × Σ|β|
```

The model must now balance two things:
1. Fit the training data well (low RSS)
2. Keep β values small (low penalty)

Result: slightly imperfect fit on training data → much better on new data.

---

## The Alpha Parameter — Full Detail

Alpha (α) is the **single most important hyperparameter** in regularization.
It controls how hard the model is punished for having large β coefficients.

### The Core Idea

```
Cost = RSS + α × Penalty

α is the volume knob on the penalty:
    Turn it up   → penalty gets louder → coefficients shrink more
    Turn it down → penalty gets quieter → coefficients stay larger
```

### What Happens at Each Alpha Value (Ridge Example)

Say you have 3 features: experience, age, education
and without regularization their coefficients are:

```
β₁ (experience) = 85000
β₂ (age)        = 72000
β₃ (education)  = 43000
```

Now watch what happens as alpha increases:

```
α = 0       β = [85000,  72000,  43000]   no penalty, same as Linear Regression
α = 0.01    β = [84100,  71300,  42700]   barely changed, tiny nudge
α = 0.1     β = [79000,  66000,  39000]   small shrinkage
α = 1       β = [61000,  49000,  28000]   moderate shrinkage, good balance
α = 10      β = [28000,  19000,   9000]   strong shrinkage
α = 100     β = [ 6000,   3800,   1500]   very strong, almost flat
α = 1000    β = [  700,    400,    150]   extreme — nearly all gone
α = 100000  β = [    7,      4,      1]   model is basically predicting the mean
```

### The Underfitting-Overfitting Dial

```
LOW alpha (0.001)              SWEET SPOT              HIGH alpha (10000)
      ↓                             ↓                         ↓
Coefficients large          Coefficients balanced       Coefficients near zero
Model too flexible           Good generalization         Model too rigid
Memorizes noise              Learns real pattern         Ignores all patterns
Overfitting ❌                    ✅                     Underfitting ❌

Train R² = 0.99              Train R² = 0.94             Train R² = 0.60
Test  R² = 0.61              Test  R² = 0.92             Test  R² = 0.58
```

### Practical Alpha Ranges to Search

```python
# For Ridge and ElasticNet — coefficients shrink but stay in model
# Search a wide logarithmic range
alphas_ridge = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# For Lasso — more sensitive, features start dropping at lower alpha
# Search smaller values too
alphas_lasso = [0.0001, 0.001, 0.01, 0.1, 1, 10]
```

Why logarithmic spacing? Because alpha's effect is exponential — the
difference between α=1 and α=10 is huge, but α=100 vs α=110 barely matters.

### Concrete Example With Code

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

print(f"{'Alpha':<10} {'Train R²':<12} {'Test R²':<12} {'Diagnosis'}")
print("-" * 55)

for alpha in alphas:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=alpha))
    ])
    pipeline.fit(X_train, y_train)

    train_r2 = pipeline.score(X_train, y_train)
    test_r2  = pipeline.score(X_test,  y_test)

    if train_r2 > 0.95 and test_r2 < 0.75:
        diagnosis = "Overfitting ❌"
    elif train_r2 < 0.70 and test_r2 < 0.70:
        diagnosis = "Underfitting ❌"
    elif abs(train_r2 - test_r2) < 0.05:
        diagnosis = "Good fit ✅"
    else:
        diagnosis = "Acceptable"

    print(f"{alpha:<10} {train_r2:<12.4f} {test_r2:<12.4f} {diagnosis}")
```

Expected output pattern:
```
Alpha      Train R²     Test R²      Diagnosis
-------------------------------------------------------
0.001      0.9920       0.6100       Overfitting ❌
0.01       0.9870       0.7800       Overfitting ❌
0.1        0.9600       0.8900       Acceptable
1          0.9400       0.9250       Good fit ✅
10         0.9100       0.9000       Good fit ✅
100        0.7800       0.7600       Acceptable
1000       0.6100       0.5900       Underfitting ❌
```

The sweet spot here is around α = 1 to 10 — train and test R² are close together and both high.

### Visual Mental Model

```
                     ← Lower alpha          Higher alpha →

Flexibility:    ████████████████████░░░░░░░░░░░░░░░░░░░
Bias:           ░░░░░░░░░░░░░░░░░░░░████████████████████
Variance:       ████████████████████░░░░░░░░░░░░░░░░░░░
Overfitting:    ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
Underfitting:   ░░░░░░░░░░░░░░░░░░░░░░░░████████████████

Sweet spot zone:            ↑↑↑↑↑↑↑↑
```

### Alpha Behaviour Differs Between Ridge and Lasso

```
Ridge with increasing alpha:
β₁: 85000 → 61000 → 28000 → 6000 → 700   (shrinks smoothly, never hits 0)
β₂: 72000 → 49000 → 19000 → 3800 → 400   (same — all survive)
β₃: 43000 → 28000 →  9000 → 1500 → 150   (same — all survive)

Lasso with increasing alpha:
β₁: 85000 → 52000 → 18000 →     0         (hits zero and stays there)
β₂: 72000 → 31000 →     0 →     0         (eliminated earlier)
β₃: 43000 →     0 →     0 →     0         (eliminated first — weakest feature)
```

Lasso eliminates the weakest features first as alpha increases.
Ridge just keeps shrinking all of them proportionally.

**Always find alpha using GridSearchCV + CV — never guess.**

---

## Ridge Regression (L2)

Penalty = sum of **squared** coefficients → `α × Σ(β²)`

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),   # always scale before Ridge
    ('model',  Ridge(alpha=1.0))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### What Ridge Does to Coefficients

```
Without Ridge:
feature1_β = 85000   (very large)
feature2_β = -72000  (very large, negative)
feature3_β = 43000

With Ridge (alpha=10):
feature1_β = 12000   (shrunk significantly)
feature2_β = -9000   (shrunk significantly)
feature3_β = 8000

All coefficients shrunk — but NONE become exactly zero.
Ridge keeps all features in the model.
```

### Key Property
```
Ridge NEVER sets any coefficient to exactly zero.
It shrinks all of them toward zero — but keeps all features.
```

| ✅ Use Ridge When | ❌ Not Ideal When |
|---|---|
| Overfitting is the problem | You want automatic feature selection |
| Multicollinearity between features | Only a few features are truly useful |
| All features seem genuinely useful | |
| Polynomial Regression with high degree | |

---

## Lasso Regression (L1)

Penalty = sum of **absolute values** of coefficients → `α × Σ|β|`

```python
from sklearn.linear_model import Lasso

pipeline = Pipeline([
    ('scaler', StandardScaler()),   # always scale before Lasso
    ('model',  Lasso(alpha=0.1))
])

pipeline.fit(X_train, y_train)

# See which features survived
print(pipeline.named_steps['model'].coef_)
# Output: [5200.0,  0.0,  3100.0,  0.0,  1800.0]
#                   ↑              ↑
#         these features were completely eliminated
```

### What Lasso Does to Coefficients

```
Without Lasso (5 features):
β = [85000, 72000, 43000, 12000, 8000]
All 5 features used in the model.

With Lasso (alpha=0.5):
β = [52000, 0, 31000, 0, 0]
Only 2 features survived — rest eliminated automatically.
```

### Why Does Lasso Eliminate Features But Ridge Doesn't?

```
Ridge penalty region = smooth circle (sphere in higher dimensions)
Lasso penalty region = diamond with sharp corners at the axes

The optimal solution almost always lands on the smooth circle (Ridge)
→ coefficients get close to zero but almost never exactly zero

The optimal solution often lands on the sharp corners of the diamond (Lasso)
→ at the corners, one or more axes = 0 exactly
→ those features get completely eliminated
```

You don't need to memorize the geometry. Just remember:

```
Ridge → shrinks all, keeps all features
Lasso → shrinks some to exactly zero, automatic feature selection
```

| ✅ Use Lasso When | ❌ Not Ideal When |
|---|---|
| Many features, suspect only some matter | All features are genuinely important |
| You want automatic feature selection | Features are highly correlated (Lasso picks one, drops others arbitrarily) |
| Interpretability is important | |

---

## ElasticNet (L1 + L2 Combined)

Combines both Ridge and Lasso penalties:

```
Cost = RSS + α × [l1_ratio × Σ|β| + (1 - l1_ratio) × Σβ²]
                   ↑ Lasso part        ↑ Ridge part
```

```python
from sklearn.linear_model import ElasticNet

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  ElasticNet(alpha=0.1, l1_ratio=0.5))
    #                                ↑
    #                   0.5 = 50% Lasso + 50% Ridge
    #                   1.0 = pure Lasso
    #                   0.0 = pure Ridge
])

pipeline.fit(X_train, y_train)
```

### The l1_ratio Parameter — Full Detail

l1_ratio controls **what mix of Lasso and Ridge** you apply inside ElasticNet.
Think of it as a slider between pure Ridge on the left and pure Lasso on the right.

```
l1_ratio = 0.0          l1_ratio = 0.5          l1_ratio = 1.0
     ↓                        ↓                        ↓
Pure Ridge             50% Ridge + 50% Lasso       Pure Lasso
Shrinks all            Shrinks + some elimination   Full elimination
No features dropped    Moderate selection           Aggressive selection
```

### What Happens at Each l1_ratio (Concrete Example)

Same dataset, same alpha, different l1_ratio:

```
Features: experience, age, education, city_score, random_noise

l1_ratio = 0.0  (pure Ridge):
β = [52000, 41000, 29000, 18000, 9000]
All 5 features survive — just shrunk

l1_ratio = 0.2  (mostly Ridge, little Lasso):
β = [51000, 40000, 27000, 14000, 0]
Weakest feature (random_noise) eliminated

l1_ratio = 0.5  (balanced):
β = [49000, 37000, 21000, 0, 0]
2 weakest features eliminated

l1_ratio = 0.8  (mostly Lasso, little Ridge):
β = [44000, 28000, 0, 0, 0]
Only 2 strongest features survive

l1_ratio = 1.0  (pure Lasso):
β = [38000, 0, 0, 0, 0]
Only the strongest feature survives
```

### How Alpha and l1_ratio Work Together

These two parameters work as a team:

```
Alpha  → controls HOW MUCH shrinkage/elimination happens (volume)
l1_ratio → controls WHAT TYPE of shrinkage (Ridge vs Lasso mix)

Example:

alpha=0.01, l1_ratio=0.9:
→ Small penalty, mostly Lasso style
→ Some features eliminated but very gently
→ Result: mild feature selection, coefficients still large

alpha=100, l1_ratio=0.1:
→ Large penalty, mostly Ridge style
→ All coefficients crushed, almost none eliminated
→ Result: very smooth model, all features barely contribute

alpha=1, l1_ratio=0.5:
→ Moderate penalty, balanced
→ Some elimination + general shrinkage
→ Result: clean, generalizable model (often the sweet spot)
```

### Visualizing the Two Parameters Together

```
                     l1_ratio →
              0.0    0.2    0.5    0.8    1.0
            ┌──────────────────────────────────┐
     0.001  │ mild  mild  mild  mild  mild     │ ← low alpha = weak effect
            │                                  │
  α  0.1    │ shrk  shrk  sel   sel   sel      │
            │                                  │
     1      │ shrk  shrk  sel   sel   sel+     │ ← moderate alpha = real effect
            │                                  │
     10     │ flat  flat  sel++ sel++ sel++    │
            │                                  │
     1000   │ zero  zero  zero  zero  zero     │ ← too high = all gone
            └──────────────────────────────────┘

shrk = all shrunk    sel = some eliminated    flat = nearly all zero
```

### Searching Both Parameters with GridSearchCV

```python
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  ElasticNet())
])

param_grid = {
    'model__alpha':    [0.001, 0.01, 0.1, 1, 10, 100],
    'model__l1_ratio': [0.1, 0.2, 0.5, 0.7, 0.9, 1.0]
}
# This tries 6 × 6 = 36 combinations
# Each combination tested with 5-fold CV = 180 fits total
# GridSearchCV handles all of this automatically

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best alpha:   ", grid_search.best_params_['model__alpha'])
print("Best l1_ratio:", grid_search.best_params_['model__l1_ratio'])
print("Best CV R²:   ", round(grid_search.best_score_, 4))
```

### Practical Starting Points

```
If you have no idea where to start:
    alpha    → try [0.01, 0.1, 1, 10, 100]
    l1_ratio → try [0.2, 0.5, 0.8]

If previous search found best alpha at the edge of your range:
    e.g. best alpha = 100 (your highest) → expand range to include 1000, 10000
    e.g. best alpha = 0.01 (your lowest) → expand range to include 0.001, 0.0001

If l1_ratio = 1.0 keeps winning → just use pure Lasso instead
If l1_ratio = 0.0 keeps winning → just use pure Ridge instead
```

### Why ElasticNet Exists

Lasso has a weakness: when features are correlated, it arbitrarily picks one and drops the rest.
Ridge handles correlated features well but doesn't do feature selection.
ElasticNet gets the best of both.

```
Correlated features: experience and age (both related to salary)

Lasso might:  keep experience, drop age entirely
Ridge might:  keep both, shrink both
ElasticNet:   keep both but shrink more aggressively than Ridge
              → feature selection + stability
```

| ✅ Use ElasticNet When | ❌ Not Ideal When |
|---|---|
| Features are correlated AND you want selection | Simple problems (overkill) |
| Unsure whether to use Ridge or Lasso | You need pure feature selection (use Lasso) |
| High-dimensional data with mixed feature importance | |

---

## Side by Side Comparison

| | Ridge (L2) | Lasso (L1) | ElasticNet |
|---|---|---|---|
| Penalty | Σβ² | Σ\|β\| | Mix of both |
| Sets β to exactly zero? | ❌ Never | ✅ Yes | ✅ Sometimes |
| Feature selection | ❌ No | ✅ Yes | ✅ Partial |
| Handles correlated features | ✅ Well | ❌ Arbitrary | ✅ Well |
| Key parameters | alpha | alpha | alpha + l1_ratio |
| Best for | Overfitting, multicollinearity | Feature selection | Both problems together |

---

## Finding Best Alpha — GridSearchCV (Full Example)

```python
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import numpy as np

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Ridge ──────────────────────────────────────────────
ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge())
])
ridge_params = {'model__alpha': [0.01, 0.1, 1, 10, 100, 1000]}
ridge_gs = GridSearchCV(ridge_pipeline, ridge_params, cv=5, scoring='r2')
ridge_gs.fit(X_train, y_train)
print("Ridge  — Best alpha:", ridge_gs.best_params_,
      "| CV R²:", round(ridge_gs.best_score_, 4))

# ── Lasso ──────────────────────────────────────────────
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Lasso())
])
lasso_params = {'model__alpha': [0.001, 0.01, 0.1, 1, 10]}
lasso_gs = GridSearchCV(lasso_pipeline, lasso_params, cv=5, scoring='r2')
lasso_gs.fit(X_train, y_train)
print("Lasso  — Best alpha:", lasso_gs.best_params_,
      "| CV R²:", round(lasso_gs.best_score_, 4))

# ── ElasticNet ─────────────────────────────────────────
en_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  ElasticNet())
])
en_params = {
    'model__alpha':    [0.01, 0.1, 1, 10],
    'model__l1_ratio': [0.2, 0.5, 0.8]
}
en_gs = GridSearchCV(en_pipeline, en_params, cv=5, scoring='r2')
en_gs.fit(X_train, y_train)
print("ElasticNet — Best params:", en_gs.best_params_,
      "| CV R²:", round(en_gs.best_score_, 4))

# ── Final evaluation on test set ───────────────────────
# Pick whichever had the best CV R²
best_model = ridge_gs.best_estimator_   # assuming Ridge won
y_pred = best_model.predict(X_test)
print("\nFinal Test R²:", r2_score(y_test, y_pred))
```

---

## The Golden Rule — Always Scale Before Regularization

```
⚠️ Never skip this step.

The penalty treats all β values equally.

salary  scale: 10,000 – 500,000  →  large β values naturally
age     scale: 1 – 100           →  small β values naturally

Without scaling, the penalty unfairly crushes age's coefficient
and barely touches salary's. The regularization becomes meaningless.

StandardScaler first → then Ridge/Lasso/ElasticNet. Always.
```

```python
# ❌ Wrong — no scaling
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ✅ Correct — always scale first
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=1.0))
])
pipeline.fit(X_train, y_train)
```

---

## Decision Guide — Which Regularization to Use?

```
Start with plain Linear Regression as a baseline.
    ↓
Is it overfitting? (Train R² >> Test R²)
    ├── No  → Keep plain Linear Regression
    └── Yes → Add regularization
                ↓
            Do you want feature selection?
            (eliminate useless features automatically)
                ├── No  → Ridge
                └── Yes → Are features correlated with each other?
                                ├── No  → Lasso
                                └── Yes → ElasticNet
```

---

## Complete Correct Workflow

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Step 1 — Split (test set locked away)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 2 — Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge())
])

# Step 3 — Use CV to find best alpha
param_grid = {'model__alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best alpha:  ", grid_search.best_params_)
print("Best CV R²:  ", round(grid_search.best_score_, 4))

# Step 4 — Final honest evaluation on test set (only once!)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Final Test R²:", round(r2_score(y_test, y_pred), 4))
```

---

## Common Mistakes

| Mistake | Why It's Wrong | Fix |
|---|---|---|
| Not scaling before regularization | Penalty unfairly affects small-scale features | Always use StandardScaler in pipeline |
| Guessing alpha | Might pick terrible value | Always use GridSearchCV + CV |
| Using CV score as final result | CV is for tuning, not reporting | Evaluate once on locked test set |
| Using Lasso with correlated features | Arbitrarily drops one of the correlated pair | Use ElasticNet instead |
| Setting alpha too high | Underfitting — model predicts near the mean | Use GridSearchCV to find the balance |
| Fitting scaler on full data | Data leakage | Always fit scaler only on X_train |

---

# PART 3 — HOW EVERYTHING CONNECTS

```
Problem: Data has a curved relationship
                ↓
Solution: Polynomial Regression (create x², x³ features)
                ↓
Problem: High degree causes overfitting
                ↓
Solution: Regularization — Ridge penalty keeps coefficients small
                ↓
Problem: How do we find best degree AND best alpha?
                ↓
Solution: Cross Validation (GridSearchCV tries all combos)
                ↓
Final: Evaluate ONCE on locked test set
```

### Polynomial + Ridge + GridSearchCV (Full Combined Example)

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: create polynomial features → scale → Ridge
pipeline = Pipeline([
    ('poly',   PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',  Ridge())
])

# Search over both degree AND alpha simultaneously
param_grid = {
    'poly__degree':  [1, 2, 3, 4, 5],
    'model__alpha':  [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(pipeline, param_grid,
                           cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best degree:", grid_search.best_params_['poly__degree'])
print("Best alpha: ", grid_search.best_params_['model__alpha'])
print("Best CV R²: ", round(grid_search.best_score_, 4))

# Final test evaluation
y_pred = grid_search.best_estimator_.predict(X_test)
print("Test R²:    ", round(r2_score(y_test, y_pred), 4))
```

---

## Master Summary Table

| Concept | What It Is | When To Use |
|---|---|---|
| Cross Validation | Splits training data K times, averages results | Always — for tuning and model comparison |
| GridSearchCV | Automates trying all hyperparameter combos using CV | Finding best alpha, degree, etc. |
| Ridge (L2) | Penalty on Σβ² — shrinks all, removes none | Overfitting, multicollinearity |
| Lasso (L1) | Penalty on Σ\|β\| — eliminates some features | Feature selection needed |
| ElasticNet | Ridge + Lasso combined | Correlated features + feature selection |
| Alpha | Controls regularization strength | Always tune via GridSearchCV |
| l1_ratio | Lasso vs Ridge balance in ElasticNet | Tune via GridSearchCV |
