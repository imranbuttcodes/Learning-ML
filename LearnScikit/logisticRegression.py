import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
"""

---

## Logistic Regression — Step 1: What is it?

First — forget the word "Regression" in the name. It's confusing! 😄

Despite the name — Logistic Regression is a **Classification** algorithm. It predicts **categories** not numbers!

---

## Linear vs Logistic

```
Linear Regression   → predicts a NUMBER
                      house price = $245,000

Logistic Regression → predicts a CATEGORY
                      email = Spam or Not Spam
```

---

## Real World Examples

```
Email       → Spam or Not Spam
Tumor       → Malignant or Benign
Student     → Pass or Fail
Transaction → Fraud or Not Fraud
Patient     → Diabetic or Not Diabetic
```

Always **two categories** — that's called **Binary Classification**! 💪

---

## The Key Difference

Linear Regression output:
```
any number → 150000, 3.14, -99, 1000000
```

Logistic Regression output:
```
always between 0 and 1 → probability!
0.92 → 92% chance of spam
0.15 → 15% chance of spam
```

Then it decides:
```
probability > 0.5 → Spam    (class 1)
probability < 0.5 → Not Spam (class 0)
```

---

"""




# hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1,  1])

# m, b = np.polyfit(hours, result, 1)
# line = [m*x + b for x in hours]

# fig, ax = plt.subplots(figsize=(8, 5))

# ax.scatter(hours, result, color="steelblue",
#            edgecolor="black", s=100, zorder=3)
# ax.plot(hours, line, color="red",
#         linewidth=2, label="Linear Regression")

# ax.set_title("Why Linear Regression Fails for Classification")
# ax.set_xlabel("Hours Studied")
# ax.set_ylabel("Result (0=Fail, 1=Pass)")
# ax.legend()
# ax.grid(linestyle="--", alpha=0.5)

# plt.tight_layout()
# plt.show()




"""
Why linear Regression failed in the above example?

No worries bro! Let me explain super simply. 🎯

---

## The Problem With Linear Regression for Classification

Look at what linear regression predicts:

```
Hours = 1  → predicts -0.07  ❌ negative!
Hours = 5  → predicts 0.47
Hours = 10 → predicts 1.07   ❌ above 1!
```

---

## Why is This a Problem?

We're trying to predict **probability** of passing:

```
0.0 = 0%  chance of passing
0.5 = 50% chance of passing
1.0 = 100% chance of passing
```

Probability can NEVER be:
```
negative  → -0.07 means -7% chance?? 😬
above 1   → 1.07 means 107% chance?? 😬
```

That makes no sense in real world! 💪

---

## Visual Problem

```
Result
  1 |         ●●●●●●
    |        ╱
    |       ╱
    |      ╱
  0 |●●●●●
    └──────────────→ Hours
         ↑        ↑
    goes below 0  goes above 1
    here ❌        here ❌
```

---

## What We Actually Need

A line that:
```
Never goes below 0   ✅
Never goes above 1   ✅
Makes a clean S curve ✅
```

Like this:

```
Result
  1 |         ●●●●●●──────
    |              ╱
    |             ╱
    |            ╱
  0 |────────●●●●
    └──────────────→ Hours
```

That S shaped curve is called **Sigmoid function** — the heart of Logistic Regression! 🔥

---


"""








# Sigmoid function
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# x = np.linspace(-10, 10, 100)
# y = sigmoid(x)

# fig, ax = plt.subplots(figsize=(8, 5))

# ax.plot(x, y, color="blue", linewidth=2.5,zorder = 3)

# # Add reference lines
# ax.axhline(y=0.5, color="red", linestyle="--",
#            alpha=0.7, label="Decision boundary (0.5)",zorder = 2)
# ax.axhline(y=1.0, color="gray", linestyle=":",
#            alpha=0.5, label="Max = 1.0",zorder = 2)
# ax.axhline(y=0.0, color="gray", linestyle=":",
#            alpha=0.5, label="Min = 0.0",zorder = 2)

# ax.set_title("Sigmoid Function", fontsize=14, fontweight="bold")
# ax.set_xlabel("Input (any number)")
# ax.set_ylabel("Output (always 0 to 1)")
# ax.legend()
# ax.grid(linestyle="--", alpha=0.5)

# plt.tight_layout()
# plt.show()



## What the Chart Shows

# Left side  → output close to 0 → Fail
# Right side → output close to 1 → Pass
# Middle     → output = 0.5      → decision boundary
# ```

# The decision boundary at 0.5 means:
# ```
# output > 0.5 → Pass (class 1)
# output < 0.5 → Fail (class 0)



# hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1,  1]) # 0 means fail and 1 means pass

# X = hours.reshape(-1,1)

# Y = result

# x_train, x_test, y_train, y_test = train_test_split (
#     X,
#     Y,
#     test_size = 0.2,
#     random_state = 42
# )

# model = LogisticRegression()

# model.fit(x_train,y_train)
# predictions = model.predict(x_test)

# predicted_probs = model.predict_proba(x_test)

# print('X_tests',x_test.flatten())
# print('Actual Values',y_test)
# print('predictions',predictions)
# print('Predictions Probabilities',predicted_probs)

"""
MOdel evaluation fro logistic regression

## Model Evaluation for Classification

Remember our regression metrics?
```
MAE, MSE, RMSE, R²  → for regression
```

For classification we use completely different metrics:
```
Accuracy, Precision, Recall, F1 Score, Confusion Matrix
```

---

## First — Accuracy

Simplest metric. Just answers:

```
Out of all predictions — how many were correct?
```

```
Accuracy = correct predictions / total predictions
```

Example:
```
Total predictions  = 10
Correct predictions = 9
Accuracy = 9/10 = 0.90 = 90%
```

---

## Calculate With Sklearn

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1,  1])

X = hours.reshape(-1, 1)
y = result

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Actual:    {y_test}")
print(f"Predicted: {predictions}")
print(f"Accuracy:  {accuracy * 100:.1f}%")
```

---

"""

hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1,  1]) # 0 means fail and 1 means pass

X = hours.reshape(-1,1)

Y = result

x_train, x_test, y_train, y_test = train_test_split (
    X,
    Y,
    test_size = 0.2,
    random_state = 42
)

model = LogisticRegression()

model.fit(x_train,y_train)
predictions = model.predict(x_test)

predicted_probs = model.predict_proba(x_test)

accuracy = accuracy_score(y_test,predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)

f1Score = f1_score(y_test, predictions)

cm = confusion_matrix(y_test,predictions)

print('X_tests',x_test.flatten())
print('Actual Values',y_test)
print('predictions',predictions)
print('Predictions Probabilities',predicted_probs)
print("Classification Evaluation")
print(f'Accuracy Score: {accuracy * 100 :.2f}')
print("Precision: ",precision)
print("Recall",recall)
print("F1Score",f1Score)
print("Confusion Matrix")
print(cm)

fig,ax = plt.subplots(figsize=(12,8))

im = ax.imshow(cm,cmap='Blues')

plt.colorbar(im)

ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(['fail','pass'])
ax.set_yticklabels(['fail','pass'])

ax.set_xlabel("Predicted", fontweight="bold", fontsize=12)
ax.set_ylabel("Actual", fontweight="bold", fontsize=12)
ax.set_title(f"Confusion Matrix\nAccuracy: {accuracy*100:.1f}%",
             fontsize=14, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(
            j,i, cm[i][j],
            ha = 'center',
            va = 'center',
            fontsize = 16,
            fontweight = 'bold',
            color = 'white' if cm[i][j] >=0.5 else 'black'
        )
plt.tight_layout()

plt.show()