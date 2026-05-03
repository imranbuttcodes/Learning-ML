# Handling Outliers — Complete Reference Guide
> ML Preprocessing Series | Imran's Study Notes

---

## What Are Outliers?

Data points that are **significantly different from the rest** — sitting far away from the main cluster of values.

```
Normal salaries:  [30k, 35k, 40k, 42k, 38k, 45k]  ← all similar range
With outlier:     [30k, 35k, 40k, 42k, 38k, 45k, 2500k]  ← CEO salary sneaked in
                                                    ↑ drags everything off
```

**Impact on models:**
- Linear Regression bends its line toward the outlier (OLS minimizes squared errors — outlier has huge squared error)
- Mean and StandardScaler get completely distorted
- KNN distance calculations get pulled away from real clusters

---

## Why Outliers Exist

```
Data Entry Error      → typed 250000 instead of 25000  → remove/correct
Measurement Error     → faulty sensor read -999°C       → remove/correct
Legitimate Extreme    → CEO salary, 7-foot athlete      → keep, handle carefully
Sampling Error        → wrong population mixed in       → investigate, remove
Rare Real Events      → Black Friday spike, earthquake  → keep — important signal
```

---

## Step 1 — Detection Methods

### Method 1 — Boxplot (Visual, Fastest)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Single column
sns.boxplot(x=df['salary'])
plt.show()

# Multiple columns
df[['age', 'salary', 'experience']].boxplot(figsize=(12, 5))
plt.show()
```

**Reading a boxplot:**

```
         ┌─────────────┐
    ─────┤             ├─────   ← whiskers (Q1 - 1.5×IQR  to  Q3 + 1.5×IQR)
         │    BOX      │        ← middle 50% of data (Q1 to Q3)
         ├─────────────┤        ← median (Q2)
    •  •                        ← outliers (beyond whiskers)

Q1  = 25th percentile
Q3  = 75th percentile
IQR = Q3 - Q1
```

---

### Method 2 — IQR Method (Most Common, Most Robust)

```python
def detect_outliers_iqr(df, column):
    Q1  = df[column].quantile(0.25)
    Q3  = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(f"Column: {column}")
    print(f"Q1={Q1}, Q3={Q3}, IQR={IQR}")
    print(f"Bounds: [{lower}, {upper}]")
    print(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)\n")

    return lower, upper

lower, upper = detect_outliers_iqr(df, 'salary')
```

**Concrete example:**
```
Salaries: [30000, 35000, 40000, 42000, 38000, 45000, 2500000]

Q1=35000, Q3=42000, IQR=7000
Lower = 35000 - 10500 = 24500
Upper = 42000 + 10500 = 52500

2,500,000 > 52,500  →  OUTLIER ✅
```

---

### Method 3 — Z-Score Method

```python
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(df['salary']))
outliers = df[z_scores > 3]   # threshold: 3 standard deviations

print(f"Outliers: {len(outliers)}")
```

**How Z-score works:**
```
Z = (value - mean) / std

Z = 0   → at the mean (normal)
Z = 1   → 1 std away (normal)
Z = 2   → 2 std away (uncommon)
Z = 3   → 3 std away → flag as outlier
Z = 5+  → extremely rare → almost certainly outlier
```

**IQR vs Z-score:**
```
IQR:     doesn't assume distribution shape → use as default
         robust for skewed data

Z-score: assumes normal distribution
         sensitive to outliers itself (mean/std get distorted)
         use only when data is normally distributed
```

---

### Method 4 — Isolation Forest (Multivariate)

Detects outliers across multiple columns simultaneously.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    contamination=0.05,   # expect ~5% outliers
    random_state=42
)

df['outlier_flag'] = iso.fit_predict(
    df[['age', 'salary', 'experience']]
)
# Returns: 1 = normal,  -1 = outlier

outliers = df[df['outlier_flag'] == -1]
print(f"Outliers detected: {len(outliers)}")
```

**Why it works:**
```
Randomly splits data using decision trees.
Outliers are isolated faster (fewer splits needed).
Short isolation path  →  outlier
Long isolation path   →  normal point
```

---

### Method 5 — Histogram (Visual)

```python
df['salary'].hist(bins=50, figsize=(10, 5))
plt.title('Salary Distribution')
plt.show()

# Long tail on one side  →  skewed + possible outliers
# Isolated bars far from cluster  →  outliers
```

---

## Step 2 — Handling Strategies

---

### Strategy 1 — Remove

Delete rows identified as outliers.

```python
Q1  = df['salary'].quantile(0.25)
Q3  = df['salary'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df[(df['salary'] >= lower) & (df['salary'] <= upper)]

print(f"Before: {len(df)} rows")
print(f"After:  {len(df_clean)} rows")
```

| ✅ Remove When | ❌ Don't Remove When |
|---|---|
| Data entry / measurement error | Outlier is legitimate extreme value |
| < 5% of data affected | Many outliers (> 5-10%) |
| Large dataset | Small dataset — can't afford to lose rows |

---

### Strategy 2 — Cap / Clip (Winsorization) ← Most Common

Cap outliers at threshold — don't remove, just limit their extreme values.

```python
# IQR-based capping
Q1  = df['salary'].quantile(0.25)
Q3  = df['salary'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['salary'] = df['salary'].clip(lower=lower, upper=upper)

# Percentile-based capping (simpler, very common)
lower_p = df['salary'].quantile(0.01)   # 1st percentile
upper_p = df['salary'].quantile(0.99)   # 99th percentile

df['salary'] = df['salary'].clip(lower=lower_p, upper=upper_p)

# Result:
# Before: [30000, 35000, 40000, 2500000]
# After:  [30000, 35000, 40000,   52500]  ← capped, not removed
```

| ✅ Cap When | ❌ Avoid When |
|---|---|
| Outlier might be legitimate | You need exact values preserved |
| Don't want to lose rows | |
| Most real-world cases | |

---

### Strategy 3 — Transform

Apply math that compresses extreme values naturally.

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer

# Log transform — most common for right-skewed data
df['salary_log'] = np.log1p(df['salary'])
# Before: [30000, 35000, 40000, 2500000]
# After:  [10.31, 10.46, 10.60,   14.73]
# Gap dramatically reduced

# Square root — gentler compression
df['salary_sqrt'] = np.sqrt(df['salary'])

# Yeo-Johnson — automatic, handles negatives too
pt = PowerTransformer(method='yeo-johnson')
df['salary_transformed'] = pt.fit_transform(df[['salary']])
```

| ✅ Transform When | ❌ Avoid When |
|---|---|
| Data is right-skewed (salary, price) | Need interpretable coefficients |
| Want to reduce skewness AND outliers together | Data has negative values (use yeo-johnson) |
| Don't want to lose any rows | |

---

### Strategy 4 — Impute

Treat outliers like missing values — replace with median.

```python
import numpy as np

Q1  = df['salary'].quantile(0.25)
Q3  = df['salary'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Flag outliers as NaN
df['salary'] = df['salary'].where(
    (df['salary'] >= lower) & (df['salary'] <= upper),
    other=np.nan
)

# Replace with median
df['salary'].fillna(df['salary'].median(), inplace=True)
```

---

### Strategy 5 — Keep + Use Robust Models

When the outlier is real and important — don't remove, just use
algorithms that aren't affected by extreme values.

```python
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

# HuberRegressor — less sensitive than OLS Linear Regression
model = HuberRegressor()

# RandomForest — tree splits unaffected by outlier scale
model = RandomForestRegressor(n_estimators=100)

# RobustScaler — uses median + IQR instead of mean + std
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

| ✅ Keep When | ❌ Don't Keep When |
|---|---|
| Outlier is legitimate and carries signal | Outlier is clearly an error |
| Rare but real event (Black Friday spike) | You're using sensitive models (OLS LR) |
| Using tree-based models | |

---

## Step 3 — Multivariate Outliers

Single-column detection misses outliers that are unusual **combinations** of values.

```
height = 180cm  ← normal alone
weight = 45kg   ← low but possible alone

Together: 180cm + 45kg = extreme combination → multivariate outlier
```

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
df['is_outlier'] = iso.fit_predict(df[['height', 'weight', 'age']])

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
df['is_outlier'] = lof.fit_predict(df[['height', 'weight', 'age']])

# -1 = outlier,  1 = normal
outliers = df[df['is_outlier'] == -1]
```

---

## The Critical Rule — Train vs Test

```
⚠️ NEVER remove or modify outliers in the test set.

The test set represents real-world data.
Real world WILL have outliers.
Removing them from test hides real model weakness.

✅ Training set → detect + handle outliers
❌ Test set     → leave completely untouched
```

---

## Complete Workflow

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1 — Visualize
df.boxplot(figsize=(12, 5))
plt.title('Before Treatment')
plt.show()

# Step 2 — Quantify
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col:<20} {n} outliers ({n/len(df)*100:.1f}%)")

# Step 3 — Handle (capping — most common choice)
for col in numeric_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower, upper=upper)

# Step 4 — Verify
df.boxplot(figsize=(12, 5))
plt.title('After Treatment')
plt.show()
```

---

## Decision Guide

```
Is the outlier a data entry / measurement error?
    └── Yes → Remove or correct it

Is it a legitimate extreme value?
    ├── Dataset is small?                → Cap (Winsorization)
    ├── Dataset is large + < 5% rows?   → Remove
    ├── Important signal?               → Keep + Robust model
    └── Data is right-skewed anyway?    → Log/Power Transform

How many outliers?
    ├── < 5%  → Safe to remove
    ├── 5-10% → Cap or Transform
    └── > 10% → Don't remove — investigate data source
```

---

## Strategy Comparison

| Strategy | Data Loss | Best For | Limitation |
|---|---|---|---|
| Remove | ✅ Yes | Data errors, < 5% outliers | May lose real data |
| Cap / Clip | ❌ No | Most real-world cases | Distorts true value |
| Log Transform | ❌ No | Right-skewed data | Changes scale |
| Impute | ❌ No | When removal not possible | Loses outlier signal |
| Keep + Robust Model | ❌ No | Legitimate extremes | Needs robust algorithm |

---

## Common Mistakes

| Mistake | Why Wrong | Fix |
|---|---|---|
| Removing outliers from test set | Hides real-world weakness | Only handle in train set |
| Always removing outliers blindly | May delete important signal | Investigate each outlier first |
| Using Z-score on skewed data | Assumes normal distribution | Use IQR for skewed data |
| Ignoring multivariate outliers | Miss unusual combinations | Use Isolation Forest |
| Not verifying after treatment | May still have outliers | Replot boxplot after |
| Removing > 10% of data | Deeper problem in data | Investigate data source |

---

## Quick Reference

```python
# Detect — IQR
Q1, Q3 = df[col].quantile([0.25, 0.75])
IQR    = Q3 - Q1
lower, upper = Q1 - 1.5*IQR,  Q3 + 1.5*IQR

# Detect — Z-score
from scipy import stats
z = np.abs(stats.zscore(df[col]))
outliers = df[z > 3]

# Remove
df = df[(df[col] >= lower) & (df[col] <= upper)]

# Cap
df[col] = df[col].clip(lower=lower, upper=upper)

# Transform
df[col] = np.log1p(df[col])

# Multivariate
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05, random_state=42)
df['outlier'] = iso.fit_predict(df[numeric_cols])
```
