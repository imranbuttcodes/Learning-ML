# Feature Encoding — Complete Reference Guide
> ML Preprocessing Series | Imran's Study Notes

---

## What is Feature Encoding?

ML models only understand **numbers**. Feature encoding converts categorical text data into numerical form so models can process it.

---

## Types of Categorical Data

| Type | Definition | Examples |
|---|---|---|
| **Nominal** | No order, just labels | Gender, City, Color |
| **Ordinal** | Has a meaningful order | Education level, Size (S/M/L), Star rating |

> ⚠️ Always identify which type you have before choosing an encoding technique.

---

## Encoding Techniques

### 1. Label Encoding
Assigns each category an integer (0, 1, 2...).

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])
# Female → 0, Male → 1
```

| ✅ Use For | ❌ Avoid For |
|---|---|
| Ordinal data | Nominal features in linear models |
| Target/output column | Any case where order doesn't exist |

> **Problem:** Model assumes Female (0) < Male (1) — implies a false relationship.

---

### 2. One-Hot Encoding (OHE)
Creates a new binary column for each category.

```python
# pandas way
pd.get_dummies(df, columns=['city'], drop_first=True)  # drop_first avoids dummy trap

# sklearn way
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = ohe.fit_transform(df[['city']])
```

| ✅ Use For | ❌ Avoid For |
|---|---|
| Nominal data | High cardinality (500 cities = 500 columns) |
| Low cardinality (< ~15 unique values) | |

> **Dummy Variable Trap:** With N categories, only N-1 columns are needed. Use `drop_first=True`.

---

### 3. Ordinal Encoding
Like label encoding but **you define the order manually**.

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['Matric', 'Inter', 'Bachelors', 'Masters']])
df['edu_encoded'] = oe.fit_transform(df[['education']])
# Matric → 0, Inter → 1, Bachelors → 2, Masters → 3
```

| ✅ Use For | ❌ Avoid For |
|---|---|
| Ordinal data with known order | Nominal data |

---

### 4. Frequency Encoding
Replaces each category with how often it appears (as a proportion).

```python
freq_map = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq_map)
# Lahore → 0.50, Karachi → 0.33, Islamabad → 0.17
```

| ✅ Use For | ⚠️ Watch Out For |
|---|---|
| High cardinality nominal data | Two cities with same frequency get same value |

---

### 5. Target Encoding
Replaces each category with the **mean of the target variable** for that category.

```python
# Safe way — fit only on training data
target_map = X_train.groupby('city')['salary'].mean()
X_train['city_encoded'] = X_train['city'].map(target_map)
X_test['city_encoded'] = X_test['city'].map(target_map)  # apply same map, never refit
```

| ✅ Use For | ❌ Never Do |
|---|---|
| High cardinality + regression problems | Fitting on full data before split (data leakage!) |

> ⚠️ **Data Leakage Risk:** Always fit target map on training data only.

---

### 6. Binary Encoding
Converts categories to integers, then to binary bits, splits each bit into a column.

```python
# pip install category_encoders
import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['city'])
df_encoded = encoder.fit_transform(df)
# 5 cities → 3 columns instead of 5 (OHE would need 5)
# 100 categories → only 7 columns (log₂100 ≈ 7)
```

| ✅ Use For | ❌ Avoid For |
|---|---|
| High cardinality nominal data | Low cardinality (OHE is simpler and more interpretable) |

---

## Quick Decision Guide

```
Is the data Ordinal (has a meaningful order)?
    ├── Yes → Ordinal Encoding (define order manually)
    └── No (Nominal)
            ├── Low cardinality (< ~15 unique values)?
            │       └── One-Hot Encoding
            └── High cardinality (> ~15 unique values)?
                    ├── Regression problem? → Target Encoding (after split!)
                    ├── Want simplicity?   → Frequency Encoding
                    └── Want compact cols? → Binary Encoding
```

---

## Summary Table

| Technique | Type | Cardinality | Creates New Cols? | Leakage Risk? |
|---|---|---|---|---|
| Label Encoding | Ordinal / Target | Any | ❌ No | ❌ No |
| One-Hot Encoding | Nominal | Low | ✅ Yes (N-1) | ❌ No |
| Ordinal Encoding | Ordinal | Any | ❌ No | ❌ No |
| Frequency Encoding | Nominal | High | ❌ No | ❌ No |
| Target Encoding | Nominal | High | ❌ No | ⚠️ Yes (if done wrong) |
| Binary Encoding | Nominal | High | ✅ Yes (log₂N) | ❌ No |

---

## The Golden Rule: Fit on Train Only

For any encoding that "learns" from data (Target Encoding, OHE with sklearn), always:

```python
encoder.fit(X_train)        # learn from training data only
encoder.transform(X_train)  # apply to train
encoder.transform(X_test)   # apply same transformation to test
```

**Never** `encoder.fit_transform(X)` on the full dataset before splitting.

---

## Production-Level Pipeline (sklearn)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

nominal_cols  = ['city', 'gender']
ordinal_cols  = ['education']
numeric_cols  = ['age', 'salary']

preprocessor = ColumnTransformer(transformers=[
    ('ohe', OneHotEncoder(handle_unknown='ignore'), nominal_cols),
    ('ord', OrdinalEncoder(categories=[['Matric','Inter','Bachelors','Masters']]), ordinal_cols),
    ('scale', StandardScaler(), numeric_cols)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)   # automatically fits encoders on train only
pipeline.predict(X_test)         # applies transformations correctly
```

> Using Pipelines is the clean, production-safe way — no manual leakage worries.

---

## Common Mistakes

| Mistake | Why It's Wrong | Fix |
|---|---|---|
| Label encoding nominal features | Implies false ordering | Use OHE instead |
| OHE on high cardinality cols | Hundreds of new columns | Use Binary or Frequency Encoding |
| Target encoding before split | Data leakage | Always fit on X_train only |
| Not using `drop_first=True` | Dummy variable trap | Add `drop_first=True` or `drop='first'` |
| Forgetting `handle_unknown='ignore'` | Crashes on unseen categories in test | Always add this to OHE |

---
