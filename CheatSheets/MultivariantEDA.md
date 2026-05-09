# Bivariate & Multivariate EDA — Study Reference

> **The step after Univariate EDA.**
> Stop inspecting features in isolation — now study *relationships* between them.

---

## Quick Recap

| EDA Type | Scope | Core Question |
|---|---|---|
| Univariate | 1 variable | What does this feature look like? |
| **Bivariate** | 2 variables | Is there a relationship between X and Y? |
| **Multivariate** | 3+ variables | How do features interact with each other? |

ML is fundamentally about learning X → Y. Bivariate and Multivariate EDA are where that learning *begins*.

---

## Bivariate EDA

### Which Tool to Use?

| Variable 1 | Variable 2 | Best Plot |
|---|---|---|
| Numerical | Numerical | Scatter plot + Correlation |
| Categorical | Numerical | Boxplot / Violin plot |
| Categorical | Categorical | Crosstab / Countplot |

---

### Case 1 — Numerical vs. Numerical

**Example:** `Gr Liv Area` vs. `SalePrice` — does bigger area mean higher price?

```python
plt.scatter(df["Gr Liv Area"], df["SalePrice"])
plt.xlabel("Ground Living Area")
plt.ylabel("Sale Price")
plt.show()
```

Each point = one row `(x_i, y_i)`.

#### Reading the Scatter Plot

| Pattern | Meaning |
|---|---|
| Upward trend ↗ | Positive relationship |
| Downward trend ↘ | Negative relationship |
| Random cloud | No relationship — feature may not help the model |
| Roughly a line | Linear — good for Linear Regression |
| Curved / arc | Nonlinear — Linear Regression will struggle |

#### Outliers
A single distant point can heavily distort Linear Regression because it minimizes *squared* errors — so huge errors dominate.

---

#### Pearson Correlation (r)

Mathematically quantifies the strength of a **linear** relationship.

```python
df["Gr Liv Area"].corr(df["SalePrice"])
# e.g. r = 0.72 → strong positive linear relationship
```

| r value | Interpretation |
|---|---|
| +1 | Perfect positive |
| 0 | No linear relationship |
| −1 | Perfect negative |

> ⚠️ **Correlation only measures linear relationships.** A curved relationship can produce r ≈ 0 even when a strong pattern exists.

---

### Case 2 — Categorical vs. Numerical

**Example:** `Neighborhood` vs. `SalePrice` — which neighborhoods have expensive houses?

```python
import seaborn as sns

sns.boxplot(x="Neighborhood", y="SalePrice", data=df)
plt.xticks(rotation=90)
plt.show()
```

Boxplot shows median, spread, and outliers **per group** — making group comparisons immediate and clear.

**Insight example:** `StoneBr` houses cluster high; `OldTown` houses cluster low → `Neighborhood` is a meaningful feature.

---

### Case 3 — Categorical vs. Categorical

**Example:** `Gender` vs. `Purchased` — are males purchasing more?

```python
pd.crosstab(df["Gender"], df["Purchased"])
```

**Example output:**

| Gender | No | Yes |
|---|---|---|
| Male | 20 | 80 |
| Female | 60 | 40 |

Patterns become immediately visible.

---

## Multivariate EDA

Real-world targets rarely depend on just one feature:

```
Y = f(X₁, X₂, X₃, ..., Xₙ)
```

Multivariate EDA studies how features relate *to each other*, not just to the target.

---

### Correlation Matrix

```python
num_df = df.select_dtypes(include="number")
corr = num_df.corr()
```

Produces a table of r values for every feature pair — reveals what's related, what's redundant, and what may cause multicollinearity.

---

### Heatmap

```python
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", annot=True)
plt.show()
```

The visual version of the correlation matrix. Strongly correlated pairs show as deep red or blue.

---

### Multicollinearity

When two features are highly correlated with *each other* (not just the target):

**Example:** `Garage Area` and `Garage Cars` — both carry nearly the same information (r ≈ 0.95).

**Why it's dangerous for Linear Regression:**
The model tries to assign credit to each feature, but if both move together, it gets confused — producing unstable, unreliable coefficients.

**Symptoms:**
- Wildly fluctuating coefficients
- Strange or contradictory weights
- Poor interpretability

**Detection:** Look for off-diagonal high-r pairs in the correlation heatmap.

---

### Pairplot

The most powerful quick-look multivariate tool for small feature subsets.

```python
sns.pairplot(df[["SalePrice", "Gr Liv Area", "Overall Qual"]])
plt.show()
```

Shows scatter plots for every feature pair plus individual distributions along the diagonal — all in one view.

---

## The Full ML Pipeline

```
1. Understand the problem       → regression or classification?
2. Load data                    → pd.read_csv()
3. Univariate EDA               → skewness, missing values, outliers, distributions
4. Bivariate EDA                → X ↔ Y relationships, linearity, useful features
5. Multivariate EDA             → feature interactions, multicollinearity, redundancy
6. Train/test split             ← stop looking at test data statistically after this
7. Preprocessing (on train only)→ imputation, scaling, encoding; then transform test
8. Train model
9. Evaluate
10. Improve                     → feature engineering, regularization, tuning
```

> **Step 6 is a hard boundary.** Fitting any imputer, scaler, or encoder on the full dataset before splitting leaks test information — **data leakage**.

---

## Key Takeaway

EDA is not about making graphs. It is about **mathematically and visually understanding your data before modeling**.

```python
# ❌ Bad ML engineer
model.fit(X, y)

# ✅ Good ML engineer
# ... 30+ minutes of EDA first, then model.fit()
```

Every preprocessing decision should trace back to something observed during EDA. If it doesn't, you're guessing.