# Logistic Regression — Complete Reference Guide
> ML Algorithms Series | Imran's Study Notes

---

## What Is It?

Despite the name — **Logistic Regression is a classification algorithm, not regression.**
It predicts the **probability** that a data point belongs to a class.

```
Input:  Study hours = 6, Sleep hours = 7
Output: P(pass) = 0.87  →  87% chance of passing
        Since 0.87 > 0.5 threshold  →  Predict: PASS
```

---

## Why Not Linear Regression for Classification?

```
Linear Regression output range: -∞ to +∞
Probability must be:             0  to  1

Linear Regression on pass/fail:
Study=1hr  → predicted = -0.2   ❌ negative, meaningless
Study=8hrs → predicted =  1.4   ❌ above 1, meaningless

Also: one outlier shifts the entire line, breaking the threshold.
```

---

## The Sigmoid Function

Squishes any value into (0, 1) — turns linear output into probability.

```
σ(z) = 1 / (1 + e^(-z))

where z = β₀ + β₁x₁ + β₂x₂ + ...  (same linear combo as Linear Regression)
```

```
Output
1.0 |          _____________
    |        /
0.5 |-------/---------------  ← decision boundary (z = 0)
    |      /
0.0 |_____/
    |________________________
    -6   -3    0    3    6
              z value

z → -∞   →  P → 0     (definitely class 0)
z =  0   →  P = 0.5   (exactly on boundary)
z → +∞   →  P → 1     (definitely class 1)
```

---

## How a Prediction Works

```
Step 1 — Calculate z:
z = β₀ + β₁×(study_hours)
z = -4  + 1.2×6  =  3.2

Step 2 — Sigmoid:
P(pass) = 1 / (1 + e^(-3.2)) = 0.96

Step 3 — Threshold (default = 0.5):
0.96 > 0.5  →  PASS ✅
```

---

## How It Learns — Log Loss

Minimizes **Log Loss** (Binary Cross Entropy), not RSS.

```
Log Loss = -1/n × Σ [y × log(ŷ) + (1-y) × log(1-ŷ)]

Actual=1, Predicted=0.95  →  tiny penalty   (confident + correct ✅)
Actual=1, Predicted=0.10  →  HUGE penalty   (confident + wrong ❌)
Actual=0, Predicted=0.02  →  tiny penalty   (correctly said no ✅)
Actual=0, Predicted=0.98  →  HUGE penalty   (very wrong ❌)
```

The more confidently wrong → the bigger the penalty.

---

## Decision Boundary

Where predicted probability = exactly 0.5 (z = 0).
Model switches from predicting class 0 to class 1 here.

```
Two features — boundary is a straight LINE in 2D space:

x₂ |    ●  ●  Class 1
   |  ●  ●
   | ─────────────────  ← decision boundary
   |  ○  ○
   |    ○  ○  Class 0
   |________________________ x₁
```

> Logistic Regression always draws a **straight line** boundary.
> For curved boundaries → add polynomial features or use a different algorithm.

---

## Full Code

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

X = df[['study_hours', 'sleep_hours']]
y = df['passed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(random_state=42))
])

pipeline.fit(X_train, y_train)

# Predict class labels
y_pred = pipeline.predict(X_test)

# Predict probabilities
y_prob = pipeline.predict_proba(X_test)
# [[0.12, 0.88],  ← 12% class 0, 88% class 1
#  [0.76, 0.24]]  ← 76% class 0, 24% class 1

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## predict vs predict_proba

```python
model.predict(X_test)
# → [1, 0, 1, 1, 0]  (class labels)

model.predict_proba(X_test)
# → [[0.12, 0.88],    col0=P(class 0), col1=P(class 1)
#    [0.76, 0.24]]    always sums to 1.0 per row
```

**When to use probabilities:**

```
Medical:       Give doctor P(cancer)=0.73 — not just yes/no
Fraud:         P>0.95 → block, P>0.6 → flag for review, P<0.3 → allow
Spam:          Adjust threshold based on cost of false positives
```

---

## Changing the Threshold

```python
y_prob = pipeline.predict_proba(X_test)[:, 1]  # P(class=1)

threshold = 0.3   # lower = more sensitive = catches more positives
y_pred_custom = (y_prob >= threshold).astype(int)
```

```
Lower threshold (0.3):
→ More predictions = class 1
→ Catches more true positives
→ Also more false positives
→ Use when missing a positive is costly (medical diagnosis)

Higher threshold (0.7):
→ Fewer predictions = class 1
→ Misses some true positives
→ But fewer false positives
→ Use when false positives are costly (spam filter)
```

---

## Key Hyperparameters

### C — Regularization Strength (inverse of alpha)

```
C = 1 / alpha

Small C  →  Strong regularization  →  simpler  →  might underfit
Large C  →  Weak regularization    →  complex  →  might overfit

C = 0.01   very strong regularization
C = 0.1    strong
C = 1.0    default — start here
C = 10     weak
C = 100    very weak
```

### penalty — Regularization Type

```
penalty='l2'          → Ridge (default) — shrinks all, keeps all features
penalty='l1'          → Lasso — eliminates some features
penalty='elasticnet'  → mix (need solver='saga')
penalty=None          → no regularization
```

### solver

```
solver='lbfgs'      → default, good for most cases
solver='liblinear'  → small datasets, supports l1
solver='saga'       → large data, supports elasticnet
```

### GridSearchCV for Hyperparameters

```python
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression())
])

param_grid = {
    'model__C':       [0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l1', 'l2'],
    'model__solver':  ['liblinear']   # liblinear supports both l1 and l2
}

grid_search = GridSearchCV(pipeline, param_grid,
                           cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV accuracy:", round(grid_search.best_score_, 4))
```

---

## Multiclass Classification

```python
# OvR — One vs Rest
# Trains N binary classifiers (one per class)
model = LogisticRegression(multi_class='ovr')

# Multinomial — Softmax
# Trains ONE model, outputs P for all classes (always sums to 1)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
```

```
3 classes: Cat, Dog, Bird

OvR trains 3 classifiers:
  Classifier 1: Cat vs (Dog + Bird)
  Classifier 2: Dog vs (Cat + Bird)
  Classifier 3: Bird vs (Cat + Dog)
  → Predict: class with highest probability

Multinomial (Softmax):
  One model outputs: P(Cat)=0.70, P(Dog)=0.20, P(Bird)=0.10
  → Predict: Cat  (always sums to 1.0)
  → More accurate, preferred when solver supports it
```

---

## Assumptions

```
1. Categorical output variable           → not for continuous targets
2. Linear decision boundary              → classes roughly separable by a line
3. No severe multicollinearity           → check VIF
4. Large enough sample size              → works poorly on < 100 rows
5. No extreme outliers                   → handle in preprocessing
6. Features should be scaled             → always StandardScaler first
```

---

## Where to Use

| ✅ Good Use Cases | ❌ Bad Use Cases |
|---|---|
| Email spam detection | Continuous output → Linear Regression |
| Disease diagnosis (sick/healthy) | Complex non-linear boundaries → Trees/SVM |
| Customer churn prediction | Image/audio classification → CNNs |
| Credit approval | Very high-dimensional data |
| Sentiment analysis (pos/neg) | |
| Any binary classification | |

---

## Pros and Cons

### ✅ Pros
| Advantage | Detail |
|---|---|
| Outputs probabilities | Gives confidence level, not just class |
| Fast to train | Even on large datasets |
| Interpretable | Coefficients show feature impact on odds |
| Great baseline classifier | Always try before complex models |
| Built-in regularization | L1/L2 via C parameter |

### ❌ Cons
| Disadvantage | Detail |
|---|---|
| Linear boundary only | Can't capture curved decision boundaries |
| Sensitive to outliers | Extreme values shift the boundary |
| Assumes linearity | Feature/log-odds must be roughly linear |
| Multicollinearity hurts it | Same issue as Linear Regression |
| Not great for complex patterns | Trees/Neural Nets handle complexity better |

---

## vs Linear Regression

| | Linear Regression | Logistic Regression |
|---|---|---|
| Task | Regression | Classification |
| Output | -∞ to +∞ | 0 to 1 (probability) |
| Core function | Linear equation | Sigmoid of linear equation |
| Loss function | RSS / MSE | Log Loss (Cross Entropy) |
| Decision boundary | None | Straight line |

---

## Decision Flow

```
Output is a category (classification task)?
    ├── No  → Linear Regression
    └── Yes → Logistic Regression (try as baseline first)
                    ↓
            Is decision boundary roughly linear?
                    ├── Yes → Logistic Regression works well
                    └── No  → Add polynomial features
                               OR switch to Decision Tree / SVM
                    ↓
            Overfitting? (high train acc, low test acc)
                    └── Reduce C (stronger regularization)
                        OR use penalty='l1' for feature selection
```

---

## Quick Reference

```python
# Basic usage
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(C=1.0, penalty='l2', random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]  # P(class=1)

# Custom threshold
y_custom = (y_prob >= 0.3).astype(int)

# Multiclass
LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Tune C and penalty
param_grid = {'model__C': [0.01, 0.1, 1, 10, 100],
              'model__penalty': ['l1', 'l2'],
              'model__solver':  ['liblinear']}
```

---

## The Non-Negotiable Rules

```
1. Always scale before Logistic Regression      → StandardScaler
2. Always try as baseline before complex models → simplest classifier
3. Use predict_proba when confidence matters    → medical, fraud, etc.
4. Tune threshold based on business cost        → not always 0.5
5. Use GridSearchCV for C and penalty           → never guess
6. Check for multicollinearity                  → VIF if many features
```
