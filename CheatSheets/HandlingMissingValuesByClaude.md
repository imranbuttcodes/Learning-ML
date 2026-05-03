# Handling Missing Values — Complete Reference Guide
> ML Preprocessing Series | Imran's Study Notes

---

## What Are Missing Values?

Missing values are cells in your dataset with no value. They appear as:

```
NaN   ← most common (Not a Number)
None  ← Python's null
NA    ← Not Available
''    ← empty string (looks present, isn't)
0     ← sometimes means missing (context dependent)
-1    ← sometimes used as missing placeholder
```

**Your model cannot process NaN. It will crash. Handle this before training — no exceptions.**

---

## Why Does Missing Data Happen?

```
Survey not answered       → person skipped a question
Data entry error          → someone forgot to fill a field
Not applicable            → "second income" for single-income household
Sensor/system failure     → IoT device offline, API call failed
Data integration issues   → two datasets merged, columns don't match
Deliberate concealment    → person didn't want to reveal income/age
```

Understanding WHY data is missing helps you decide HOW to handle it.

---

## Step 1 — Detecting Missing Values

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Total missing per column
print(df.isnull().sum())

# Percentage missing per column
print(df.isnull().sum() / len(df) * 100)

# Overall missing in entire dataset
print(df.isnull().sum().sum())           # total missing cells
print(df.isnull().mean().mean() * 100)   # overall % missing

# Which rows have missing values
print(df[df.isnull().any(axis=1)])

# Heatmap — visual overview
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False,
            yticklabels=False, cmap='viridis')
plt.title('Missing Value Heatmap')
# Yellow = missing, Purple = present
plt.show()
```

---

## Step 2 — The Three Types of Missing Data

**Most important concept — type of missingness decides your strategy.**

### MCAR — Missing Completely At Random

Missingness has **no relationship** to any data. Pure random chance.

```
Example: Survey system randomly crashes and loses some responses.
The crash has nothing to do with who answered or what they said.

Test: Does missing in Age correlate with any other column?
→ No correlation anywhere → MCAR
```

**Impact:** Safe to drop or use simple imputation. Least problematic.

---

### MAR — Missing At Random

Missingness is related to **other observed columns** but NOT to the missing value itself.

```
Example: Men are less likely to answer the income question.
Income is missing more often for men — but missingness is
related to GENDER (observed), not to the income value itself.

Gender:  Male   Female  Male   Female  Male
Income:  NaN    60000   NaN    45000   NaN
```

**Impact:** Handle well with imputation that uses other columns (KNN, Iterative).

---

### MNAR — Missing Not At Random

Missingness is related to the **value itself** — data is missing BECAUSE of what it would have been.

```
Example: Very high earners skip the income question deliberately.
Very low earners skip it out of embarrassment.
The NaN values ARE the extreme incomes.
```

**Impact:** Hardest to handle. Simple imputation introduces bias.
Use domain knowledge + flag missingness as a feature.

---

### Quick Identification

```
"Why is this value missing?"

Pure accident / random system failure     → MCAR → drop or simple impute
Because of another column I can see      → MAR  → use other columns in imputation
Because of what the value would have been → MNAR → domain knowledge + flag column
```

---

## Step 3 — All Strategies

---

### Strategy 1 — Dropping

Remove rows or columns with missing values.

```python
# Drop rows with ANY missing value
df_dropped = df.dropna()

# Drop rows where SPECIFIC columns have missing values
df_dropped = df.dropna(subset=['Age', 'Salary'])

# Drop columns with too many missing values (> 50%)
threshold = 0.5
df_dropped = df.loc[:, df.isnull().mean() < threshold]

# Drop columns where ALL values are missing
df_dropped = df.dropna(axis=1, how='all')
```

| ✅ Drop Rows When | ❌ Don't Drop When |
|---|---|
| MCAR — truly random missingness | MAR or MNAR — introduces bias |
| < 5% of rows affected | Many rows affected — lose too much data |
| Large dataset — losing few rows is fine | Small dataset |

| ✅ Drop Columns When | ❌ Don't Drop When |
|---|---|
| > 50-70% values missing | Column is a key feature |
| Column not important for prediction | < 50% missing — impute instead |

---

### Strategy 2 — Mean / Median / Mode Imputation

Replace missing values with a summary statistic.

```python
from sklearn.impute import SimpleImputer

# Mean — for normally distributed numerical columns
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# Median — for skewed data or columns with outliers
imputer = SimpleImputer(strategy='median')
df['Salary'] = imputer.fit_transform(df[['Salary']])

# Mode — for categorical columns
imputer = SimpleImputer(strategy='most_frequent')
df['City'] = imputer.fit_transform(df[['City']])

# Constant — fill with a fixed value
imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
df['City'] = imputer.fit_transform(df[['City']])
```

**Mean vs Median:**

```
Mean:
→ Normal distribution (symmetric bell curve)
→ Sensitive to outliers
→ Use for: Age, height, temperature

Median:
→ Skewed distributions or outliers present
→ Not affected by extreme values
→ Use for: Salary, house prices, income
```

**Limitation:**
```
Reduces variance — pulls everything toward the center
Ignores relationships between columns
Simple but not very accurate
```

---

### Strategy 3 — KNN Imputation

Finds K most similar rows and uses their values to estimate the missing one.

```python
from sklearn.impute import KNNImputer

# Always scale before KNN imputation
from sklearn.preprocessing import StandardScaler

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(df)

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(X_scaled),
    columns=df.columns
)
```

**How it works:**

```
Missing row:  Age=NaN, Salary=60000, Experience=5

KNN finds 5 most similar rows by Salary + Experience:
Row A: Age=32, Salary=58000, Experience=5
Row B: Age=29, Salary=62000, Experience=4
Row C: Age=34, Salary=57000, Experience=6
Row D: Age=31, Salary=63000, Experience=5
Row E: Age=30, Salary=59000, Experience=5

Average Age of neighbors = 31.2
→ Fill missing Age with 31.2
```

| ✅ Use KNN When | ❌ Avoid When |
|---|---|
| MAR — missingness related to other cols | Very large datasets (slow) |
| Want to preserve feature relationships | You forget to scale first |
| Medium-sized datasets | |

---

### Strategy 4 — Iterative Imputation (Most Powerful)

Treats each column with missing values as a regression problem.
Uses ALL other columns to predict missing values. Repeats until stable.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

imputer = IterativeImputer(
    estimator=LinearRegression(),
    max_iter=10,       # repeat 10 rounds
    random_state=42
)

df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)
```

**How it works:**

```
Round 1:
→ Fill all missing with mean (rough start)
→ Use Age + Experience + City to predict missing Salary
→ Use Salary + Experience + City to predict missing Age

Round 2:
→ Use Round 1 estimates (better than mean)
→ Re-predict missing Salary with improved Age estimates
→ Re-predict missing Age with improved Salary estimates

Rounds 3...10:
→ Each round gets more accurate
→ Values converge and stabilize
```

| ✅ Use Iterative When | ❌ Avoid When |
|---|---|
| Complex relationships between columns | Very large datasets (very slow) |
| MAR data | Categorical columns (use encoding first) |
| Accuracy matters more than speed | |

---

### Strategy 5 — Forward / Backward Fill

Only for **time series** — fill with previous or next known value.

```python
# Forward fill — use previous value
df['temperature'] = df['temperature'].fillna(method='ffill')

# Backward fill — use next value
df['temperature'] = df['temperature'].fillna(method='bfill')

# Example:
# Before: [22, NaN, NaN, 25, NaN, 27]
# ffill:  [22,  22,  22, 25,  25, 27]
# bfill:  [22,  25,  25, 25,  27, 27]
```

| ✅ Use When | ❌ Avoid When |
|---|---|
| Time series data (temperature, stock, sensor) | Non-time series data |
| Temporal patterns exist | No meaningful order in rows |

---

### Strategy 6 — Flag Missingness as a Feature (Pro Tip)

For MNAR data — the fact that something IS missing is itself information.
Create a binary indicator column BEFORE imputing.

```python
# Step 1 — Create indicator columns
df['age_was_missing']    = df['age'].isnull().astype(int)
df['salary_was_missing'] = df['salary'].isnull().astype(int)

# Step 2 — Then impute normally
df['age'].fillna(df['age'].median(), inplace=True)
df['salary'].fillna(df['salary'].median(), inplace=True)

# Now model knows WHERE values were missing
# AND has reasonable fill values
# Useful when the pattern of missingness predicts the target
```

---

## Strategy Comparison

| Strategy | Best For | Handles MAR? | Speed | Accuracy |
|---|---|---|---|---|
| Drop rows | MCAR, < 5% missing | ❌ No | ⚡ Fast | N/A |
| Drop columns | > 50% missing | ❌ No | ⚡ Fast | N/A |
| Mean imputation | Numeric, normal dist | ❌ No | ⚡ Fast | Low |
| Median imputation | Numeric, skewed/outliers | ❌ No | ⚡ Fast | Low |
| Mode imputation | Categorical | ❌ No | ⚡ Fast | Low |
| KNN imputation | Numeric, medium data | ✅ Yes | 🐢 Slow | Medium |
| Iterative imputation | Numeric, complex data | ✅ Yes | 🐢 Slow | High |
| Forward/Backward fill | Time series only | ✅ Yes | ⚡ Fast | High |

---

## Decision Guide

```
How much data is missing per column?
    ├── > 60-70% missing → Drop the column entirely
    └── < 60% missing   → Impute
                ↓
        What type of column?
            ├── Categorical → Mode or Constant ('Unknown')
            └── Numerical
                        ↓
                What type of missingness?
                    ├── MCAR + < 5% rows  → Drop the rows
                    ├── MCAR + more rows  → Mean or Median
                    ├── MAR (related to other columns)
                    │       ├── Small/medium dataset → Iterative Imputer
                    │       └── Large dataset        → KNN Imputer
                    └── MNAR → Flag as feature first
                                Then impute with Median/Mode
```

---

## Correct Pipeline Integration

**Always fit imputer on training data only — never on full dataset.**

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_cols     = ['age', 'salary', 'experience']
categorical_cols = ['city', 'gender']

# Numeric: impute then scale
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

# Categorical: impute then encode
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline,     numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model',        LogisticRegression())
])

# Fits imputers on X_train only — automatically
full_pipeline.fit(X_train, y_train)
full_pipeline.predict(X_test)
```

---

## Common Mistakes

| Mistake | Why It's Wrong | Fix |
|---|---|---|
| Imputing before train-test split | Leaks test set statistics into train | Always split first, impute in pipeline |
| Using mean for skewed data | Pulls values toward wrong center | Use median for skewed columns |
| Using mean/median for categorical | Can't take mean of 'Lahore' | Use mode or constant |
| Dropping rows for MAR/MNAR data | Introduces sampling bias | Use KNN or Iterative imputer |
| Forgetting to scale before KNN | KNN uses distance — scale matters | StandardScaler before KNNImputer |
| Dropping column with 30% missing | Losing useful information | Impute instead |
| Ignoring MNAR patterns | Model misses information in missingness | Flag missingness as binary feature |

---

## Quick Reference

```python
# Detect
df.isnull().sum()
df.isnull().sum() / len(df) * 100

# Simple imputation
SimpleImputer(strategy='mean')         # numeric, normal dist
SimpleImputer(strategy='median')       # numeric, skewed/outliers
SimpleImputer(strategy='most_frequent') # categorical
SimpleImputer(strategy='constant', fill_value='Unknown')

# Advanced imputation
KNNImputer(n_neighbors=5)             # use other columns, medium data
IterativeImputer(max_iter=10)         # most accurate, slow

# Time series
df.fillna(method='ffill')             # forward fill
df.fillna(method='bfill')             # backward fill

# Flag missingness
df['col_was_missing'] = df['col'].isnull().astype(int)
```