# 🚀 Pandas Cheat Sheet (Complete Guide)

A structured, beginner-to-intermediate guide to Pandas with clear syntax, explanations, and examples.

---

# 📌 1. Import Pandas

```python
import pandas as pd
import numpy as np
```

---

# 📌 2. Core Data Structures

## 🔹 Series (1D)

```python
s = pd.Series([10, 20, 30])
print(s)
```

👉 Single column of data with index

---

## 🔹 DataFrame (2D Table)

```python
data = {
    "name": ["Ali", "Ahmed", "Sara"],
    "age": [20, 22, 19],
    "marks": [85, 90, 78]
}

df = pd.DataFrame(data)
print(df)
```

👉 Rows + Columns (like Excel table)

---

# 📌 3. Basic Data Inspection

```python
df.head()       # first 5 rows

df.tail()       # last 5 rows

df.shape        # (rows, columns)

df.columns      # column names

df.info()       # data types + null info

df.describe()   # statistical summary
```

---

# 📌 4. Selecting Data

## 🔹 Select Column

```python
df["name"]
```

👉 Returns Series

```python
df[["name"]]
```

👉 Returns DataFrame

---

## 🔹 Select Multiple Columns

```python
df[["name", "marks"]]
```

---

## 🔹 Select Rows (iloc)

👉 Position-based

```python
df.iloc[0]        # first row

df.iloc[0:2]      # rows 0 to 1

df.iloc[0, 2]     # row 0, column 2
```

---

## 🔹 Select Rows (loc)

👉 Label-based

```python
df.loc[0]         # first row

df.loc[0:2]       # includes 2 (IMPORTANT)

df.loc[0, "name"]
```

---

# 📌 5. Filtering Data

## 🔹 Single Condition

```python
df[df["marks"] > 80]
```

---

## 🔹 Multiple Conditions

```python
df[(df["marks"] > 80) & (df["age"] < 22)]
```

```python
df[(df["marks"] > 85) | (df["age"] < 20)]
```

⚠️ Always use parentheses

---

# 📌 6. Adding / Updating Columns

```python
df["grade"] = ["A", "A+", "B"]
```

```python
df["marks"] = df["marks"] + 5
```

---

# 📌 7. Dropping Data

## 🔹 Drop Column

```python
df.drop("age", axis=1)
```

## 🔹 Drop Row

```python
df.drop(0, axis=0)
```

## 🔹 Inplace Modification

```python
df.drop("age", axis=1, inplace=True)
```

---

# 📌 8. Handling Missing Data (NaN)

## 🔹 Detect Missing Values

```python
df.isnull()

df.isnull().sum()
```

---

## 🔹 Drop Missing Values

```python
df.dropna()
```

```python
df.dropna(axis=1)
```

---

## 🔹 Fill Missing Values

```python
df.fillna(0)
```

```python
df["marks"] = df["marks"].fillna(50)
```

```python
df["age"] = df["age"].fillna(df["age"].mean())
```

---

# 📌 9. Aggregation Functions

```python
df["marks"].sum()
df["marks"].mean()
df["marks"].max()
df["marks"].min()
df["marks"].count()
```

---

# 📌 10. GroupBy (VERY IMPORTANT)

## 🔹 Basic Grouping

```python
df.groupby("dept")
```

👉 Returns GroupBy object

---

## 🔹 Apply Aggregation

```python
df.groupby("dept")["marks"].mean()
```

```python
df.groupby("dept")["marks"].sum()
```

```python
df.groupby("dept")["marks"].count()
```

---

## 🔹 Multiple Groups

```python
df.groupby(["dept", "name"])["marks"].mean()
```

---

## 🔹 View Groups

```python
for name, group in df.groupby("dept"):
    print(name)
    print(group)
```

---

# 📌 11. Sorting Data

```python
df.sort_values("marks")

df.sort_values("marks", ascending=False)
```

---

# 📌 12. Useful Attributes

```python
df.index

df.values

df.dtypes
```

---

# 📌 13. Key Mental Model

👉 Pandas workflow:

```
Load Data → Inspect → Clean → Filter → Analyze → Group → Output
```

---

# 🧠 Core Concepts Summary

* Series = 1D column
* DataFrame = 2D table
* iloc = index-based selection
* loc = label-based selection
* Filtering = boolean conditions
* GroupBy = split → apply → combine
* NaN = missing data

---

# 🚀 You Are Now Ready For:

👉 Real CSV datasets
👉 Data cleaning pipelines
👉 Machine Learning preprocessing
👉 Feature engineering
