# Feature Scaling — Complete Reference Guide
> ML Preprocessing Series | Imran's Study Notes

---

## What is Feature Scaling?

Feature scaling brings all numerical features to a **comparable range** so no single feature dominates due to its unit or magnitude — without changing the information the feature carries.

---

## Why Does Scale Matter?

```python
age    = [22, 35, 45]              # range ~100
salary = [30000, 80000, 150000]    # range ~150,000

# KNN calculates distance — salary dominates, age becomes irrelevant
# even if age is actually more important for the prediction
```

---

## Which Models Need Scaling?

| Needs Scaling ✅ | Doesn't Need Scaling ❌ |
|---|---|
| Linear Regression (with regularization) | Decision Trees |
| Logistic Regression | Random Forest |
| KNN | XGBoost / LightGBM |
| SVM | Gradient Boosting |
| Neural Networks | Naive Bayes |
| PCA | |
| K-Means | |

---

## Scaling Techniques

### 1. Standardization (Z-score)
**Formula:** `z = (x - mean) / std` → Output: mean=0, std=1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

| ✅ Use For | ❌ Avoid When |
|---|---|
| Most ML models (safe default) | You need a strict [0,1] range |
| When outliers exist | |

---

### 2. Min-Max Normalization
**Formula:** `x_scaled = (x - min) / (max - min)` → Output: [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()                        # default [0, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))  # custom range

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

| ✅ Use For | ❌ Avoid When |
|---|---|
| Neural Networks | Data has significant outliers |
| Image pixel data (0–255 → 0–1) | |

---

### 3. Robust Scaling
**Formula:** `x_scaled = (x - median) / IQR` → Output: unbounded, centered on median

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

> Uses IQR (Q3 - Q1) — middle 50% of data. Outliers don't influence the scale.

| ✅ Use For | ❌ Avoid When |
|---|---|
| Data with outliers you can't remove | You need fixed output range |
| Noisy real-world datasets | |

---

### 4. Max Abs Scaling
**Formula:** `x_scaled = x / |x_max|` → Output: [-1, 1]

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

| ✅ Use For | ❌ Avoid When |
|---|---|
| Sparse data (lots of zeros) | Data has outliers |
| TF-IDF / text features | |

> Doesn't center data — preserves sparsity (zeros stay zero).

---

### 5. Log Transformation
**Formula:** `x_scaled = log(1 + x)` → Compresses large values, expands small ones

```python
import numpy as np

df['salary_log'] = np.log1p(df['salary'])  # log1p handles zeros safely

# Check if needed:
df['salary'].skew()  # > 1 or < -1 → apply transformation
```

| ✅ Use For | ❌ Avoid When |
|---|---|
| Right-skewed distributions | Negative values (use Yeo-Johnson instead) |
| Income, house prices, population | Data is already normally distributed |

---

### 6. Power Transformation (Box-Cox / Yeo-Johnson)
Automatically finds the best power to make distribution as normal as possible.

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')  # works on negative + positive
pt = PowerTransformer(method='box-cox')      # positive values only

X_train_scaled = pt.fit_transform(X_train)
X_test_scaled  = pt.transform(X_test)
```

| ✅ Use For | ❌ Avoid When |
|---|---|
| Heavily skewed data | You need interpretability |
| When log transform isn't enough | |

> **Default pick:** Yeo-Johnson — it handles negatives too.

---

## Quick Decision Guide

```
Does your data have significant outliers?
    ├── Yes → Robust Scaler
    └── No
            ├── Is data heavily skewed?
            │       ├── Yes → Log Transform or Power Transform → then StandardScaler
            │       └── No
            │               ├── Neural Network / need fixed [0,1]?
            │               │       └── MinMax Scaler
            │               ├── Sparse data (lots of zeros)?
            │               │       └── MaxAbs Scaler
            │               └── Everything else?
            │                       └── Standard Scaler ← safe default
```

---

## Standardization vs Normalization

| | StandardScaler | MinMaxScaler |
|---|---|---|
| Output range | Unbounded (mean=0, std=1) | Fixed [0, 1] |
| Outlier sensitivity | Low ✅ | High ⚠️ |
| Preserves shape | No | Yes |
| Best for | Most ML models | Neural nets, images |

> **When in doubt → use StandardScaler.**

---

## The Golden Rule — Fit on Train Only

```python
scaler.fit(X_train)            # learn stats from training data only
scaler.transform(X_train)      # apply to train
scaler.transform(X_test)       # apply SAME stats to test — never refit!
```

> Fitting on full data leaks test set statistics into training → **data leakage**.

---

## Common Mistakes

| Mistake | Why It's Wrong | Fix |
|---|---|---|
| `fit_transform` on full dataset before split | Data leakage | Split first, fit only on train |
| MinMax on outlier-heavy data | Outliers compress everything | Use RobustScaler |
| Scaling tree-based models | Wastes time, no benefit | Skip scaling for trees |
| Not scaling before PCA | PCA is variance-based, large scale = dominant | Always scale before PCA |
| Forgetting to scale test set | Train/test on different scales | Always `transform(X_test)` |

---

## Production Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

numeric_cols     = ['age', 'salary']
image_cols       = ['pixel_intensity']
categorical_cols = ['city']

preprocessor = ColumnTransformer(transformers=[
    ('std',    StandardScaler(),                          numeric_cols),
    ('minmax', MinMaxScaler(),                            image_cols),
    ('ohe',    OneHotEncoder(handle_unknown='ignore'),    categorical_cols)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model',        LogisticRegression())
])

pipeline.fit(X_train, y_train)   # fits scalers on train only automatically
pipeline.predict(X_test)
```

---

## Summary Table

| Technique | Output Range | Outlier Safe | Sparse Safe | Best For |
|---|---|---|---|---|
| StandardScaler | Unbounded | ✅ Moderate | ❌ | Most models (default) |
| MinMaxScaler | [0, 1] | ❌ | ❌ | Neural nets, images |
| RobustScaler | Unbounded | ✅ Best | ❌ | Outlier-heavy data |
| MaxAbsScaler | [-1, 1] | ❌ | ✅ | Sparse / TF-IDF data |
| Log Transform | Varies | ✅ | ❌ | Right-skewed data |
| PowerTransformer | ~Normal | ✅ | ❌ | Heavily skewed data |

---
