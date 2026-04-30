# 🚀 NumPy Cheat Sheet (From Basics to ML)

This cheat sheet covers everything you learned step-by-step — from basic arrays to ML-level operations.

---

# 📌 1. Import NumPy

```python
import numpy as np
```

---

# 📌 2. Creating Arrays

```python
# 1D Array
a = np.array([1, 2, 3])

# 2D Array
b = np.array([[1, 2], [3, 4]])

# Special arrays
np.zeros((2,3))
np.ones((2,2))
np.arange(0,10,2)
np.linspace(0,1,5)
```

---

# 📌 3. Basic Operations

```python
a = np.array([1,2,3])

a + 5      # [6,7,8]
a * 2      # [2,4,6]
a ** 2     # [1,4,9]
```

👉 Element-wise operations

---

# 📌 4. Shape & Dimensions

```python
a.shape     # shape

a.ndim      # number of dimensions

a.size      # total elements
```

---

# 📌 5. Indexing & Slicing

## 1D

```python
a = np.array([10,20,30,40,50])

a[0]        # 10
a[-1]       # 50

a[1:4]      # [20,30,40]
a[:3]       # [10,20,30]
a[::2]      # [10,30,50]
a[::-1]     # reverse
```

## 2D

```python
b = np.array([
 [1,2,3],
 [4,5,6],
 [7,8,9]
])

b[0,0]      # 1
b[1,2]      # 6

b[0]        # first row
b[:,1]      # second column

b[1:3,1:3]  # submatrix
```

⚠️ NumPy slicing returns a VIEW

```python
slice_a = a[1:3]
slice_a[0] = 100
# original array changes
```

---

# 📌 6. Broadcasting

```python
a = np.array([1,2,3])

a + 5   # [6,7,8]
```

```python
matrix = np.array([
 [1,2,3],
 [4,5,6]
])

vector = np.array([10,20,30])

matrix + vector
```

👉 Applied row-wise automatically

---

# 📌 7. Boolean Masking

```python
a = np.array([10,20,30,40,50])

a > 25
# [False, False, True, True, True]

# Filtering
a[a > 25]
# [30,40,50]
```

## Combining Conditions

```python
a[(a > 10) & (a < 50)]
```

## Modify Values

```python
a[a > 30] = 0
```

---

# 📌 8. Aggregation Functions

```python
a = np.array([10,20,30,40])

np.sum(a)
np.mean(a)
np.max(a)
np.min(a)
```

---

# 📌 9. Axis (VERY IMPORTANT)

```python
b = np.array([
 [1,2,3],
 [4,5,6]
])
```

## axis = 0 (column-wise)

```python
np.sum(b, axis=0)
# [5,7,9]
```

## axis = 1 (row-wise)

```python
np.sum(b, axis=1)
# [6,15]
```

👉 axis = direction you collapse

---

# 📌 10. Reshaping

```python
a = np.array([1,2,3,4,5,6])

a.reshape(2,3)

a.reshape(2,-1)
```

⚠️ Total elements must match

---

# 📌 11. Flattening

```python
b.flatten()   # copy
b.ravel()     # view
```

---

# 📌 12. Row vs Column Vector

```python
a = np.array([1,2,3])

a.reshape(1,3)   # row

a.reshape(3,1)   # column
```

---

# 📌 13. Transpose

```python
b.T
```

---

# 📌 14. Dot Product (ML Core)

```python
x = np.array([1,2,3])
w = np.array([4,5,6])

np.dot(x, w)
# 32

x @ w
```

---

# 📌 15. Matrix Multiplication

```python
A @ B
```

⚠️ Columns of A = Rows of B

---

# 📌 16. Normalization

## Min-Max

```python
(data - np.min(data)) / (np.max(data) - np.min(data))
```

## Column-wise

```python
(data - np.min(data, axis=0)) / \
(np.max(data, axis=0) - np.min(data, axis=0))
```

---

# 📌 17. Mini ML Example

```python
X = np.array([
 [1,2],
 [3,4],
 [5,6]
])

W = np.array([0.5, 1.0])

predictions = X @ W
```

---

# 🧠 Final Summary

NumPy gives you:

* Fast arrays
* Mathematical operations
* Data manipulation
* ML foundation

---

# 🔥 What You Can Do Now

* Handle datasets
* Perform vectorized operations
* Build ML logic (linear models)

---

You are now ready for:
👉 Gradient Descent
👉 Linear Regression
👉 Real ML Projects
