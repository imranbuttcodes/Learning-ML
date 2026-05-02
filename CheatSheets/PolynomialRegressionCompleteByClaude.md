# Polynomial Regression & Multiple Polynomial Regression
# Complete Reference Guide
> ML Algorithms Series | Imran's Study Notes

---

# PART 1 — WHY POLYNOMIAL REGRESSION EXISTS

---

## The Problem With Linear Regression

Linear Regression draws a straight line through data:

```
y = β₀ + β₁x
```

This works perfectly when the relationship between x and y is linear —
meaning y grows (or shrinks) at a constant rate as x increases.

```
LINEAR relationship (straight line fits perfectly):

Experience:  1     2     3     4     5
Salary:    30k   35k   40k   45k   50k
           ↑ grows by exactly 5k every year

Plot:
50k |                         *
45k |                    *
40k |               *
35k |          *
30k |     *
    |________________________
      1    2    3    4    5
```

But what about this:

```
NON-LINEAR relationship (straight line fails badly):

Experience:  1     2     3     4     5     6     7
Salary:    20k   22k   27k   35k   46k   60k   78k
           ↑ jumps get BIGGER each year — accelerating

Plot:
78k |                                   *
60k |                              *
46k |                         *
35k |                    *
27k |               *
22k |          *
20k |     *
    |________________________________
      1    2    3    4    5    6    7
```

A straight line through this data is wrong almost everywhere:

```
78k |                                   * (actual)
    |                         /← straight line too low here
46k |                    *   /
    |               *   /
    |    line too  /
22k |  high here→ *
20k |     *
    |________________________________
```

The model systematically underpredicts at the extremes.
This is called UNDERFITTING — the model is too simple for the data.

This is exactly the problem Polynomial Regression solves.

---

# PART 2 — WHAT IS POLYNOMIAL REGRESSION

---

## The Core Idea

Instead of fitting a straight line, fit a **curve** by adding
**higher-degree power terms** of the existing feature.

```
Linear Regression (degree 1):
y = β₀ + β₁x                          → straight line

Polynomial Regression (degree 2):
y = β₀ + β₁x + β₂x²                  → parabola (U shape or ∩)

Polynomial Regression (degree 3):
y = β₀ + β₁x + β₂x² + β₃x³          → S-curve (one bend)

Polynomial Regression (degree n):
y = β₀ + β₁x + β₂x² + ... + βₙxⁿ    → n-1 bends
```

You are NOT adding new features from the dataset.
You are creating **mathematical transformations** of the existing feature
and handing them all to LinearRegression to fit.

---

## Why Is It Still Called "Linear" Regression?

This confuses everyone. Clear answer:

The word **"linear"** refers to how the **β coefficients** appear in the
equation — each one is just multiplied once, never raised to a power.

```
y = β₀ + β₁x + β₂x²

β₀ → appears once, just added              ✅ linear in β
β₁ → multiplied by x once                 ✅ linear in β
β₂ → multiplied by x² once                ✅ linear in β

x² is non-linear in x — but that's a FEATURE, not a coefficient.
The model treats x² as just another column of numbers.
```

So sklearn's LinearRegression fits Polynomial Regression perfectly —
you just transform the features first using PolynomialFeatures.

```python
# The model doesn't know or care that this column is x²
# It just sees a number and fits a β to it
```

---

## What Does "Degree" Mean?

Degree = the highest power of x you create.

```
Degree 1 → y = β₀ + β₁x
           Straight line. No curves.

Degree 2 → y = β₀ + β₁x + β₂x²
           Parabola. One curve direction.
           Opens up (U) or opens down (∩) depending on sign of β₂.

Degree 3 → y = β₀ + β₁x + β₂x² + β₃x³
           S-curve. Can bend in two directions.

Degree 4 → y = β₀ + β₁x + β₂x² + β₃x³ + β₄x⁴
           Two humps. More complex shape.
```

Visual shapes:

```
Degree 1          Degree 2          Degree 3          Degree 4
(straight)        (parabola)        (S-curve)         (two humps)

    /                 *                  *               *   *
   /               *     *            *               *       *
  /              *         *        *               *           *
 /                           *   *
/                               *
```

Rule: **degree - 1 = number of bends in the curve**
- Degree 2 → 1 bend
- Degree 3 → 2 bends
- Degree 10 → 9 bends (dangerously wiggly)

---

# PART 3 — HOW IT WORKS INTERNALLY

---

## Step by Step Walkthrough

**Original data:**
```
x = [1,  2,  3,  4,  5]
y = [2,  8, 18, 32, 50]
```

**Step 1 — Apply PolynomialFeatures(degree=2, include_bias=False)**

Creates new columns x and x²:

```
Original x    →    Transformed [x, x²]

x = 1         →    [1,   1]
x = 2         →    [2,   4]
x = 3         →    [3,   9]
x = 4         →    [4,  16]
x = 5         →    [5,  25]
               col1  col2
                x    x²
```

**Step 2 — Feed into LinearRegression**

Now LinearRegression sees two features (x and x²) and fits:

```
y = β₀ + β₁×(col1) + β₂×(col2)
y = β₀ + β₁×x      + β₂×x²
```

It finds β₀, β₁, β₂ that minimize RSS — same OLS as always.
The result is a curved line through the data.

**Step 3 — The model can now capture the curve**

```python
# What actually happens inside sklearn:
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_transformed = poly.fit_transform(X)

print(X_transformed)
# [[1,  1],
#  [2,  4],
#  [3,  9],
#  [4, 16],
#  [5, 25]]

# LinearRegression then fits on this transformed X
```

---

## The include_bias Parameter

```python
PolynomialFeatures(degree=2, include_bias=True)   # default
# Output: [1, x, x²]
# The column of 1s = bias term (same as intercept β₀)
# BUT LinearRegression already adds intercept automatically
# So having it twice causes problems

PolynomialFeatures(degree=2, include_bias=False)  # recommended
# Output: [x, x²]
# No duplicate intercept — cleaner and correct
```

**Always use `include_bias=False` when pairing with LinearRegression.**

---

# PART 4 — SIMPLE POLYNOMIAL REGRESSION (ONE FEATURE)

---

## Full Code

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# Data — clearly non-linear
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 8, 18, 32, 50, 72, 98, 128, 162, 200])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print("R²:  ", round(r2_score(y_test, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))

# Visualize fit
X_plot = np.linspace(1, 10, 200).reshape(-1, 1)
y_plot = pipeline.predict(X_plot)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', zorder=5, label='Actual data')
plt.plot(X_plot, y_plot, color='red', label='Polynomial fit (degree=2)')
plt.xlabel('X (Experience)')
plt.ylabel('y (Salary)')
plt.title('Polynomial Regression — Degree 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Why Use a Pipeline?

```python
# Without pipeline — dangerous, easy to make mistakes:
poly   = PolynomialFeatures(degree=2, include_bias=False)
scaler = StandardScaler()

X_train_poly   = poly.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_poly)

X_test_poly    = poly.transform(X_test)          # easy to forget transform vs fit_transform
X_test_scaled  = scaler.transform(X_test_poly)   # easy to accidentally refit on test

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# With pipeline — clean, automatic, safe:
pipeline = Pipeline([
    ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])

pipeline.fit(X_train, y_train)      # fits poly + scaler on train only automatically
pipeline.predict(X_test)            # transforms test correctly automatically
```

Pipeline automatically:
- Fits transformers only on training data
- Applies same transformations to test data
- Prevents data leakage
- Makes GridSearchCV work seamlessly

---

# PART 5 — CHOOSING THE RIGHT DEGREE

---

## The Most Critical Decision

Degree controls everything — too low underfits, too high overfits.
This is the single most important choice in Polynomial Regression.

## Underfitting vs Overfitting vs Good Fit

```
UNDERFITTING (degree too low):

Actual data:  *  *  *  *  *  *  *   (curved pattern)
Degree 1:    ——————————————————————  (straight line, misses curve)

Train R² = 0.72  (bad)
Test  R² = 0.69  (bad)
Both low — model too simple

─────────────────────────────────────────────────────

GOOD FIT (degree just right):

Actual data:  *  *  *  *  *  *  *
Degree 2:     ∪ smooth curve fits through pattern ∪

Train R² = 0.97  (good)
Test  R² = 0.95  (good)
Both high and CLOSE to each other ✅

─────────────────────────────────────────────────────

OVERFITTING (degree too high):

Actual data:  *  *  *  *  *  *  *
Degree 15:    ~∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿~   (wiggles through every point)

Train R² = 1.00  (perfect — too perfect, suspicious)
Test  R² = 0.12  (terrible on new data)
Gap is huge — model memorized noise
```

## Diagnosing By Comparing Train vs Test R²

```
Train R² high + Test R² high + small gap  →  Good fit ✅
Train R² high + Test R² low  + big gap   →  Overfitting ❌  → lower degree or add Ridge
Train R² low  + Test R² low  + small gap →  Underfitting ❌ → higher degree
```

---

## Finding Best Degree — Cross Validation

Never guess the degree. Use CV to find it scientifically.

```python
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

degrees = range(1, 11)

print(f"{'Degree':<8} {'Train R²':<12} {'CV R²':<12} {'CV Std':<10} {'Diagnosis'}")
print("─" * 65)

for d in degrees:
    pipeline = Pipeline([
        ('poly',   PolynomialFeatures(degree=d, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model',  LinearRegression())
    ])

    # Get both train and CV scores
    results = cross_validate(pipeline, X_train, y_train,
                             cv=5, scoring='r2',
                             return_train_score=True)

    train_r2 = results['train_score'].mean()
    cv_r2    = results['test_score'].mean()
    cv_std   = results['test_score'].std()

    if train_r2 > 0.95 and cv_r2 < 0.80:
        diagnosis = "Overfitting ❌"
    elif train_r2 < 0.75:
        diagnosis = "Underfitting ❌"
    elif abs(train_r2 - cv_r2) < 0.05:
        diagnosis = "Good fit ✅"
    else:
        diagnosis = "Acceptable"

    print(f"{d:<8} {train_r2:<12.4f} {cv_r2:<12.4f} {cv_std:<10.4f} {diagnosis}")

# Pick the degree with highest CV R²
```

Expected output pattern:

```
Degree   Train R²     CV R²        CV Std     Diagnosis
─────────────────────────────────────────────────────────────────
1        0.7200       0.6900       0.0312     Underfitting ❌
2        0.9750       0.9600       0.0180     Good fit ✅
3        0.9820       0.9550       0.0210     Good fit ✅
4        0.9910       0.9200       0.0380     Acceptable
5        0.9980       0.8400       0.0720     Overfitting ❌
8        0.9999       0.4100       0.1800     Overfitting ❌
10       1.0000       -0.320       0.4200     Overfitting ❌
```

Best degree here = 2 (highest CV R², small std, small gap from train R²).

---

## Using GridSearchCV to Find Best Degree

```python
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('poly',  PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

param_grid = {'poly__degree': [1, 2, 3, 4, 5, 6]}

grid_search = GridSearchCV(pipeline, param_grid,
                           cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best degree:", grid_search.best_params_['poly__degree'])
print("Best CV R²: ", round(grid_search.best_score_, 4))

# Final evaluation
y_pred = grid_search.best_estimator_.predict(X_test)
print("Test R²:    ", round(r2_score(y_test, y_pred), 4))
```

---

# PART 6 — BIAS-VARIANCE TRADEOFF

---

## The Most Important Theoretical Concept in ML

Polynomial Regression demonstrates this better than any other algorithm.

## What is Bias?

Error from **wrong assumptions** — model is too simple to capture the pattern.

```
High bias model (degree 1 on curved data):
- Always wrong in the same direction
- Straight line constantly misses the curve
- Doesn't matter how much data you give it — still wrong
- Called UNDERFITTING
```

## What is Variance?

Sensitivity to **small changes in training data** — model is too flexible.

```
High variance model (degree 15):
- Tiny change in one training point → completely different curve
- Perfectly fits THIS training set
- Falls apart on any other data
- Called OVERFITTING
```

## The Tradeoff

```
You cannot have low bias AND low variance at the same time.
Reducing one increases the other. Always.

LOW DEGREE (degree 1-2):
    Bias:     HIGH  ████████████████░░░░░░░░
    Variance: LOW   ████░░░░░░░░░░░░░░░░░░░░
    Result:   Underfitting

HIGH DEGREE (degree 10+):
    Bias:     LOW   ████░░░░░░░░░░░░░░░░░░░░
    Variance: HIGH  ████████████████████████
    Result:   Overfitting

SWEET SPOT (degree 2-4 typically):
    Bias:     MEDIUM  ████████░░░░░░░░░░░░░░
    Variance: MEDIUM  ████████░░░░░░░░░░░░░░
    Result:   Good generalization ✅
```

## Real Numbers Example

```
            Train R²    Test R²    Gap      Bias    Variance
Degree 1:   0.72        0.69       0.03     High    Low      ← underfit
Degree 2:   0.97        0.95       0.02     Low     Low      ← sweet spot ✅
Degree 3:   0.98        0.93       0.05     Low     Medium   ← still ok
Degree 5:   0.999       0.84       0.159    Low     High     ← overfit starts
Degree 10:  1.000       0.12       0.88     None    Extreme  ← badly overfit
Degree 15:  1.000       -2.10      3.10     None    Extreme  ← completely useless
```

The gap between Train R² and Test R² is your variance indicator.
The absolute level of Test R² is your combined bias+variance indicator.

---

# PART 7 — OVERFITTING AND HOW TO FIX IT

---

## Why High Degree Causes Overfitting

With degree 15 on 20 training points, the model has MORE parameters than needed.

```
20 data points, degree 15 → 15 β coefficients to fit
The model can contort itself to pass through every single point exactly
Including the noisy, random ones
Result: perfect training score, terrible test score
```

## Fix 1 — Lower the Degree

Simplest fix. Use CV to find the right degree.

## Fix 2 — Ridge Regularization (Most Common Fix)

Add a penalty for large β coefficients. Keeps the curve smooth
even at high degrees.

```python
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ('poly',   PolynomialFeatures(degree=5, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=1.0))   # penalty keeps β values small
])
```

What Ridge does to a high-degree polynomial:

```
Without Ridge (degree 5, overfit):
β = [85000, -72000, 43000, -31000, 18000]
    ↑ huge values → wild, wiggly curve

With Ridge (degree 5, alpha=10):
β = [12000,  -8000,  4500,  -2100,   900]
    ↑ much smaller → smooth, sensible curve
```

## Fix 3 — Find Both Degree and Alpha Together (Best Approach)

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('poly',   PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',  Ridge())
])

param_grid = {
    'poly__degree': [1, 2, 3, 4, 5],
    'model__alpha': [0.01, 0.1, 1, 10, 100]
}
# 5 × 5 = 25 combinations
# 25 × 5 folds = 125 total fits

grid_search = GridSearchCV(pipeline, param_grid,
                           cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best degree:", grid_search.best_params_['poly__degree'])
print("Best alpha: ", grid_search.best_params_['model__alpha'])
print("Best CV R²: ", round(grid_search.best_score_, 4))

# Final evaluation — only once on test set
y_pred = grid_search.best_estimator_.predict(X_test)
print("Test R²:    ", round(r2_score(y_test, y_pred), 4))
```

---

# PART 8 — MULTIPLE POLYNOMIAL REGRESSION

---

## What Is It?

Simple Polynomial Regression = one feature, add its powers.
Multiple Polynomial Regression = multiple features, add their powers
AND their interaction terms.

```
Simple (1 feature, degree 2):
y = β₀ + β₁x + β₂x²

Multiple (2 features, degree 2):
y = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₁x₂ + β₅x₂²
                                   ↑
                          interaction term — NEW concept
```

---

## What Are Interaction Terms?

An interaction term is the product of two features: `x₁ × x₂`

It captures something that neither x₁ nor x₂ alone can capture:
**how two features work TOGETHER to affect the output.**

### Real World Example

```
Predicting house price with:
x₁ = Size (sq ft)
x₂ = Location score (1-10)

Without interaction term:
Price = β₀ + β₁×Size + β₂×Location
→ Size contributes the same amount regardless of location
→ Location contributes the same amount regardless of size
→ This is WRONG in reality

With interaction term:
Price = β₀ + β₁×Size + β₂×Location + β₃×(Size×Location)
→ A large house in a prime location is worth WAY more
  than just (large house value) + (prime location value)
→ The combination is synergistic
→ β₃ captures exactly this "together they're worth more" effect
```

Another example:

```
Predicting crop yield with:
x₁ = Fertilizer amount
x₂ = Water amount

Fertilizer alone:  some yield
Water alone:       some yield
Fertilizer + Water TOGETHER: dramatically more yield
→ Their interaction is what actually drives the outcome
→ x₁×x₂ captures this perfectly
```

---

## What PolynomialFeatures Creates With Multiple Features

### With 2 features (x₁, x₂), degree=2:

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[2, 3]])   # x₁=2, x₂=3

poly = PolynomialFeatures(degree=2, include_bias=False)
result = poly.fit_transform(X)

print(result)
# [[2, 3, 4, 6, 9]]
#   x₁ x₂ x₁² x₁x₂ x₂²

print(poly.get_feature_names_out(['x1', 'x2']))
# ['x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']
```

Output columns: `[x₁, x₂, x₁², x₁x₂, x₂²]` → 5 columns

### With 2 features (x₁, x₂), degree=3:

```
[x₁, x₂, x₁², x₁x₂, x₂², x₁³, x₁²x₂, x₁x₂², x₂³] → 9 columns
```

### Formula for total columns:

```
Total columns = C(n + d, d) - 1

where:
n = number of original features
d = degree

Examples:
n=1, d=2 → C(3,2) - 1 = 3 - 1 = 2      columns: [x, x²]
n=2, d=2 → C(4,2) - 1 = 6 - 1 = 5      columns: [x₁, x₂, x₁², x₁x₂, x₂²]
n=2, d=3 → C(5,3) - 1 = 10 - 1 = 9
n=3, d=2 → C(5,2) - 1 = 10 - 1 = 9
n=5, d=3 → C(8,3) - 1 = 56 - 1 = 55
n=10,d=3 → C(13,3)- 1 = 286 - 1 = 285  ← memory issues!
```

---

## Feature Explosion Problem

As features and degree grow, columns multiply explosively:

```
Features    Degree    Output Columns
1           2         2
1           3         3
2           2         5
2           3         9
3           2         9
3           3         19
5           2         20
5           3         55
10          2         65
10          3         285        ← slow
20          2         230
20          3         1770       ← serious memory issues
50          2         1325
100         2         5150       ← impractical
```

This is called the **Curse of Dimensionality** — too many columns relative
to data points causes overfitting and memory/compute issues.

---

## Fixes for Feature Explosion

### Fix 1 — interaction_only=True

Only creates interaction terms (x₁x₂) — skips powers (x₁², x₂²).

```python
PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# For x₁, x₂:
# Normal degree=2:            [x₁, x₂, x₁², x₁x₂, x₂²]  → 5 cols
# interaction_only=True:      [x₁, x₂, x₁x₂]             → 3 cols

# Use when you care about feature interactions but not polynomial curves
```

### Fix 2 — Reduce Degree

```python
# degree=2 is almost always enough for real-world data
# degree=3 occasionally useful
# degree=4+ rarely needed, almost always overfits
```

### Fix 3 — Feature Selection Before Polynomial

```python
# Select only your most important features first
# Then apply PolynomialFeatures
# Fewer input features = dramatically fewer output columns

from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(f_regression, k=5)   # keep top 5 features
X_selected = selector.fit_transform(X_train, y_train)
# Now apply poly on only 5 features instead of 20
```

### Fix 4 — Use Tree-Based Models Instead

```python
# When you have many features and complex relationships:
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# These handle non-linearity and interactions automatically
# No feature explosion, no manual degree selection needed
```

---

## Full Code — Multiple Polynomial Regression

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# Dataset — multiple features
data = {
    'size':       [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000, 3500],
    'bedrooms':   [2,   2,    3,    3,    3,    4,    4,    4,    5,    5   ],
    'location':   [5,   6,    5,    7,    6,    8,    7,    9,    8,    10  ],
    'price':      [120, 150,  160,  200,  210,  280,  270,  350,  400,  500 ]
}

df = pd.DataFrame(data)
X = df[['size', 'bedrooms', 'location']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check how many columns different degrees create
for degree in [1, 2, 3]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_transformed = poly.fit_transform(X_train)
    print(f"Degree {degree}: {X_train.shape[1]} features → "
          f"{X_transformed.shape[1]} columns")
    print(f"  Feature names: {list(poly.get_feature_names_out(X.columns))}\n")

# Build pipeline with Ridge (recommended for multiple features)
pipeline = Pipeline([
    ('poly',   PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',  Ridge())
])

# Search degree AND alpha together
param_grid = {
    'poly__degree': [1, 2, 3],
    'model__alpha': [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(pipeline, param_grid,
                           cv=5, scoring='r2',
                           verbose=1)
grid_search.fit(X_train, y_train)

print("\nBest degree:", grid_search.best_params_['poly__degree'])
print("Best alpha: ", grid_search.best_params_['model__alpha'])
print("Best CV R²: ", round(grid_search.best_score_, 4))

# Inspect what features were created with best degree
best_degree = grid_search.best_params_['poly__degree']
poly_inspect = PolynomialFeatures(degree=best_degree, include_bias=False)
poly_inspect.fit_transform(X_train)
print("\nFeatures created:")
for name in poly_inspect.get_feature_names_out(X.columns):
    print(f"  {name}")

# Final evaluation on test set
y_pred = grid_search.best_estimator_.predict(X_test)
print("\nFinal Test R²:  ", round(r2_score(y_test, y_pred), 4))
print("Final Test RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
```

---

## Inspecting Which Features Actually Matter

After fitting, you can see which polynomial features the model found important:

```python
# Get the fitted Ridge model
ridge_model = grid_search.best_estimator_.named_steps['model']
poly_step   = grid_search.best_estimator_.named_steps['poly']

# Get feature names
feature_names = poly_step.get_feature_names_out(X.columns)

# Get coefficients
coefficients = ridge_model.coef_

# Show sorted by importance
importance = pd.DataFrame({
    'feature':     feature_names,
    'coefficient': coefficients
}).sort_values('coefficient', key=abs, ascending=False)

print(importance.to_string(index=False))
```

Output might look like:

```
      feature    coefficient
  size location     8450.23   ← interaction: size × location is strongest
      size^2        6120.45   ← quadratic size effect
  size bedrooms     4830.12   ← size × bedrooms interaction
         size       3920.67
     location       2140.33
     bedrooms        890.21
   location^2        420.55
  bedrooms^2          95.33   ← barely matters
```

This tells you which polynomial terms are actually driving the predictions.

---

## interaction_only vs Full Polynomial — When to Use Which

```
Full PolynomialFeatures (default):
    Creates: x₁, x₂, x₁², x₁x₂, x₂²
    Use when: relationship might be curved (quadratic) AND interactive
    Example: price = f(size², size×location, location²)

interaction_only=True:
    Creates: x₁, x₂, x₁x₂  (no powers)
    Use when: features interact but relationship isn't curved
    Example: crop yield = f(fertilizer × water) but not f(fertilizer²)
    Also use when: you want to limit feature explosion

Linear only (degree=1):
    Creates: x₁, x₂
    Use when: relationship is truly linear
    Always try this as baseline first
```

---

# PART 9 — ASSUMPTIONS

---

## Inherited From Linear Regression (All Still Apply)

| # | Assumption | How to Check |
|---|---|---|
| 1 | **Linearity in coefficients** | Always holds — still linear in β |
| 2 | **Homoscedasticity** | Plot residuals vs predicted — should be random cloud |
| 3 | **Normality of residuals** | Q-Q plot |
| 4 | **No autocorrelation** | Durbin-Watson test (~2.0 = good) |
| 5 | **No multicollinearity** | VIF score |

## Extra Concern — Multicollinearity Gets Worse

```python
# x and x² are ALWAYS correlated with each other
# x, x², x³ are all correlated with each other

# This is expected and unavoidable in Polynomial Regression
# Solution: Ridge regularization handles this automatically
# Don't panic about VIF for polynomial terms — just use Ridge

from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ('poly',   PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=1.0))   # handles multicollinearity
])
```

## Checking Residuals (Important)

```python
pipeline.fit(X_train, y_train)
y_pred_train = pipeline.predict(X_train)
residuals = y_train - y_pred_train

import matplotlib.pyplot as plt

# Plot 1: Residuals vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_pred_train, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
# Should look like a random cloud around 0
# Funnel shape = heteroscedasticity problem

# Plot 2: Q-Q plot for normality
import scipy.stats as stats
plt.subplot(1, 2, 2)
stats.probplot(residuals, dist='norm', plot=plt)
plt.title('Q-Q Plot')
# Points should follow the diagonal line
plt.tight_layout()
plt.show()
```

---

# PART 10 — PROS, CONS, AND WHEN TO USE

---

## Pros

| Advantage | Detail |
|---|---|
| Captures non-linear patterns | Major step up from straight-line Linear Regression |
| Uses LinearRegression internally | Fast, mathematically well-understood |
| Interpretable at low degrees | Degree 2-3 coefficients still explainable |
| Captures feature interactions | x₁x₂ term finds synergies between features |
| Easy sklearn implementation | Just add PolynomialFeatures to pipeline |
| Works great for physics/engineering | Many real relationships are polynomial |
| Can combine with Ridge | High flexibility without overfitting |

## Cons

| Disadvantage | Detail |
|---|---|
| Feature explosion | Columns multiply fast — 10 features × degree 3 = 285 columns |
| Overfits easily | Especially at high degrees without regularization |
| Multicollinearity | x and x² always correlated — mitigate with Ridge |
| Degree selection required | Must use CV every time — no universal best degree |
| Extrapolation is dangerous | Predictions far outside training range go wild |
| Sensitive to outliers | More so than plain Linear Regression |
| Hard to interpret at high degrees | degree 8+ coefficients are meaningless to explain |

---

## When to Use Polynomial Regression

```
✅ Good use cases:
- Salary vs experience (accelerating growth)
- Drug dosage vs effect (grows then plateaus)
- Speed vs fuel consumption (quadratic relationship)
- Physics problems (projectile motion, thermodynamics)
- Economic growth curves
- Any problem where scatter plot shows a clear curve

❌ Bad use cases:
- Very high-dimensional data → use XGBoost / Random Forest
- Complex image/text patterns → use Neural Networks
- When interpretability is critical → high degree is uninterpretable
- When extrapolation is needed → polynomial curves explode outside training range
- When you have 20+ features → feature explosion is unmanageable
```

---

# PART 11 — COMPLETE DECISION FLOW

---

```
STEP 1: Plot your data (scatter plot)
    ├── Looks linear? → Use Linear Regression
    └── Looks curved? → Polynomial Regression

STEP 2: Start with degree=2
    Use CV to check if it actually helps vs degree=1

STEP 3: How many features?
    ├── 1-5 features → full PolynomialFeatures(degree=2 or 3)
    ├── 5-15 features → use interaction_only=True or degree=2 only
    └── 15+ features → consider Random Forest / XGBoost instead

STEP 4: Is it overfitting? (Train R² >> Test R²)
    ├── Yes → Add Ridge regularization
    └── No  → Keep LinearRegression

STEP 5: Find best degree + alpha using GridSearchCV + CV

STEP 6: Evaluate ONCE on locked test set

STEP 7: Still not good enough?
    └── Switch to Random Forest, XGBoost, or Neural Network
```

---

# PART 12 — COMPLETE QUICK REFERENCE

---

## Key Parameters

| Parameter | What It Does | Recommended |
|---|---|---|
| `degree` | Highest power to create | Start with 2, use CV to tune |
| `include_bias` | Adds column of 1s | Always False with LinearRegression |
| `interaction_only` | Only x₁x₂, no x₁², x₂² | True when features are many |
| `alpha` (Ridge) | Regularization strength | Tune via GridSearchCV |

## Column Count Formula

```
C(n + d, d) - 1    where n = features, d = degree
```

## Bias-Variance Summary

```
Low degree  → High Bias,    Low Variance  → Underfitting
High degree → Low Bias,     High Variance → Overfitting
Sweet spot  → Balanced      Balanced      → Generalization ✅
```

## The Non-Negotiable Rules

```
1. Always use Pipeline                  → prevents leakage, clean code
2. Always scale before regularization   → StandardScaler in pipeline
3. Always use CV for degree selection   → never guess
4. Always use Ridge at degree 3+        → prevents overfitting
5. Always evaluate test set only once   → at the very end
6. Always start with degree=2           → lowest complexity that captures curves
7. Always compare to Linear Regression  → baseline before adding complexity
```

## Complete Template (Copy-Paste Ready)

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# 1. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Pipeline
pipeline = Pipeline([
    ('poly',   PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('model',  Ridge())
])

# 3. GridSearchCV
param_grid = {
    'poly__degree': [1, 2, 3, 4, 5],
    'model__alpha': [0.01, 0.1, 1, 10, 100]
}
grid_search = GridSearchCV(pipeline, param_grid,
                           cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# 4. Results
print("Best degree:", grid_search.best_params_['poly__degree'])
print("Best alpha: ", grid_search.best_params_['model__alpha'])
print("Best CV R²: ", round(grid_search.best_score_, 4))

# 5. Final evaluation (only once)
y_pred = grid_search.best_estimator_.predict(X_test)
print("Test R²:    ", round(r2_score(y_test, y_pred), 4))
print("Test RMSE:  ", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
```
