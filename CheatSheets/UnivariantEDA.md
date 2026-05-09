# Univariate EDA — Study Reference

> **Exploratory Data Analysis, one feature at a time.**
> The first real layer of understanding any dataset — before models, before preprocessing, before feature engineering.

---

## What is Univariate EDA?

**Uni** = one · **Variate** = variable/feature/column

Univariate EDA means analyzing **one feature at a time** to understand its shape, spread, quirks, and problems.

---

## Why Does It Matter?

Every column has its own personality. Without inspecting each one individually, preprocessing choices become guesswork. Univariate EDA makes every decision **logical and justified**.

| Observation | Preprocessing Action |
|---|---|
| Positive skew | Log transform |
| Extreme outliers | Use median (not mean) |
| Different scales | Standardization |
| Missing categorical values | Mode imputation or `"None"` |
| High missing % | Drop the feature |
| Rare categories | Combine or drop |
| ID / useless column | Remove entirely |

---

## Numerical Features

### Key Questions
- What is the range? Min/max realistic?
- Is it skewed? Which direction?
- Does it have outliers?
- Any missing values?
- Symmetric or lopsided?

### Basic Statistics
```python
df["SalePrice"].describe()
# Returns: count, mean, std, min, max, quartiles
```

### Mean vs. Median

| Data Type | Use |
|---|---|
| Symmetric / normal | **Mean** — represents center well |
| Skewed / has outliers | **Median** — robust to extremes |

```
[10, 12, 13, 14, 1000]
  Mean   → 209.8  ← destroyed by outlier
  Median → 13     ← stable
```

---

## Distribution & Skewness

A distribution describes **how values are spread**.

```python
plt.hist(df["SalePrice"], bins=30)   # histogram
df["SalePrice"].skew()               # skewness score
```

| Skew Value | Shape |
|---|---|
| ≈ 0 | Symmetric / normal |
| > 0 | Positive skew — long right tail |
| < 0 | Negative skew — long left tail |

**Examples of positively skewed real-world data:** income, house prices, population counts.

### Why Skewness Matters
Many ML algorithms and statistical tests assume near-normal data. Skewed features increase variance, destabilize learning, and make optimization harder.

---

## Log Transformation

Compresses extreme values and brings skewed data closer to a normal distribution.

```python
import numpy as np
np.log(df["SalePrice"])
```

> ⚠️ **Log transform ≠ StandardScaler**
> `StandardScaler` changes *scale* but **not shape** — skewness remains.
> Log transform changes the actual **distribution shape**.

---

## Boxplot & Outliers

```python
plt.boxplot(df["SalePrice"])
```

| Component | Meaning |
|---|---|
| Q1 | 25th percentile |
| Q2 | Median (50th percentile) |
| Q3 | 75th percentile |
| IQR | Q3 − Q1 |

**Outlier boundaries:**

```
Lower fence: Q1 − 1.5 × IQR
Upper fence: Q3 + 1.5 × IQR
```

Outliers are especially dangerous for **linear regression** — they pull the regression line and distort coefficients.

---

## Missing Values

```python
df.isnull().sum()
```

### What's Safe Before Train/Test Split?
Structural decisions that don't use data statistics:
- Dropping columns with 95%+ missing values
- Removing ID columns
- Removing obviously useless columns

### What MUST Happen After Splitting?
Anything that learns from statistics (mean, median, std, mode) — because applying these to the full dataset **leaks test information**.

> This is called **Data Leakage** — one of the most dangerous mistakes in ML.

---

## Categorical Features

```python
df["Neighborhood"].value_counts()
sns.countplot(x="Neighborhood", data=df)
```

### Key Questions
- Which category dominates?
- Are classes balanced?
- Any rare categories?
- Any typo duplicates? (`"Male"`, `"male"`, `"M"`)

Rare categories can hurt model generalization. Class imbalance affects learning. Encoding strategy depends on these observations.

---

## The Full Picture

```
RAW DATASET
    ↓
Univariate EDA        ← inspect each feature individually
    ↓
Informed Decisions    ← transform, scale, impute, encode
    ↓
Clean Dataset
    ↓
Modeling
```

Later stages — **Bivariate** and **Multivariate EDA** — study relationships *between* features. Univariate EDA is the essential foundation that comes first.

---

*Univariate EDA is like a medical examination of each organ separately — before you study how the organs work together.*