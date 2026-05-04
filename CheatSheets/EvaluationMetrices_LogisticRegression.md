# Classification Metrics — Complete Reference Guide
> ML Evaluation Series | Imran's Study Notes

---

## Why Accuracy Alone Fails

```
Dataset: 995 healthy patients + 5 cancer patients

Model that predicts EVERYONE as healthy:
Accuracy = 995/1000 = 99.5%  ← looks incredible

But it missed ALL 5 cancer patients.
Completely useless — yet accuracy looks great.
```

This is the **accuracy paradox**. We need metrics that capture WHAT KIND of mistakes the model makes — not just how many.

---

## The Confusion Matrix — Foundation of Everything

```
                    PREDICTED
                  Positive  Negative
         ┌────────────────────────────┐
ACTUAL   │    TP      │     FN        │
Positive │  (correct) │   (missed)    │
         ├────────────┼───────────────┤
ACTUAL   │    FP      │     TN        │
Negative │ (false     │   (correct)   │
         │  alarm)    │               │
         └────────────────────────────┘
```

```
TP — True Positive:   Predicted +, Actually +  → Hit ✅
TN — True Negative:   Predicted -, Actually -  → Correct rejection ✅
FP — False Positive:  Predicted +, Actually -  → False alarm ❌ (Type I Error)
FN — False Negative:  Predicted -, Actually +  → Miss ❌ (Type II Error)
```

**FN is usually the most dangerous mistake** — telling a sick person they're healthy.

## Concrete Example

```
10 patients — cancer screening:
Actual:    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
Predicted: [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]

Patient 1: TP ✅   Patient 2: TP ✅   Patient 3: FN ❌
Patient 4: FP ❌   Patient 5-10: TN ✅

Confusion Matrix:
              Predicted+  Predicted-
Actual+    [     2      |     1     ]   TP=2, FN=1
Actual-    [     1      |     6     ]   FP=1, TN=6
```

## Code

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_actual, y_predicted)
print(cm)
# [[TN, FP],
#  [FN, TP]]

tn, fp, fn, tp = cm.ravel()

# Visual
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.show()
```

---

## 1. Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_actual, y_predicted)
```

| ✅ Use When | ❌ Avoid When |
|---|---|
| Classes are balanced | Imbalanced datasets |
| Quick sanity check | Cancer/fraud/spam detection |

---

## 2. Precision

**"Of all I predicted positive — how many actually were?"**

```
Precision = TP / (TP + FP)

Our example: 2 / (2+1) = 0.667 = 66.7%
Out of 3 flagged as cancer — 2 actually had it.
```

```python
from sklearn.metrics import precision_score
precision_score(y_actual, y_predicted)
```

| ✅ Prioritize When | Example |
|---|---|
| False positives are costly | Spam filter (blocking real emails) |
| | Legal system (convicting innocent) |

---

## 3. Recall (Sensitivity / True Positive Rate)

**"Of all actual positives — how many did I catch?"**

```
Recall = TP / (TP + FN)

Our example: 2 / (2+1) = 0.667 = 66.7%
Out of 3 actual cancer patients — caught 2, missed 1.
```

```python
from sklearn.metrics import recall_score
recall_score(y_actual, y_predicted)
```

| ✅ Prioritize When | Example |
|---|---|
| False negatives are costly | Cancer detection (missing sick patient) |
| | Fraud detection (missing fraud costs money) |
| | COVID testing (missing infected spreads disease) |

---

## The Precision-Recall Tradeoff

**You cannot maximize both simultaneously.**

```
Lower threshold → Higher Recall, Lower Precision
                  (catch more positives, more false alarms too)

Higher threshold → Lower Recall, Higher Precision
                   (miss more positives, but fewer false alarms)

Threshold = the dial between these two extremes.
```

```
Example — 5 patients, model probabilities:
Patient:      A     B     C     D     E
P(cancer):  0.95  0.80  0.60  0.40  0.20
Actual:        1     1     0     1     0

Threshold=0.5: TP=2, FP=1, FN=1 → Precision=0.67, Recall=0.67
Threshold=0.3: TP=3, FP=1, FN=0 → Precision=0.75, Recall=1.00 ← catches all
Threshold=0.9: TP=1, FP=0, FN=2 → Precision=1.00, Recall=0.33 ← many missed
```

---

## 4. F1 Score

**Harmonic mean of Precision and Recall — one balanced number.**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Why harmonic mean (not regular average)?**

```
Precision=1.0, Recall=0.0 (useless model):
Regular avg = (1.0 + 0.0) / 2 = 0.5  ← misleadingly okay
Harmonic    = 2×(1.0×0.0)/(1.0+0.0) = 0.0  ← correctly shows useless

F1 = 0 when either P or R = 0
F1 = 1 when both P and R = 1
F1 is heavily penalized by whichever of P/R is lower
```

```python
from sklearn.metrics import f1_score
f1_score(y_actual, y_predicted)
# For multiclass:
f1_score(y_actual, y_predicted, average='macro')    # equal weight per class
f1_score(y_actual, y_predicted, average='weighted') # weighted by class size
f1_score(y_actual, y_predicted, average='micro')    # global TP/FP/FN
```

| ✅ Use F1 When |
|---|
| Imbalanced datasets |
| You need one number balancing P and R |
| Standard metric for most classification problems |

---


10. Macro, Weighted, and Micro Averaging (Multiclass Metrics)

When dealing with multiclass classification, metrics like Precision, Recall, and F1 are computed per class.
So we need a way to combine them into a single overall score.

Macro Average

"Treat every class equally — regardless of size."

Macro Precision = (P₁ + P₂ + ... + Pₙ) / n
Macro Recall    = (R₁ + R₂ + ... + Rₙ) / n
Macro F1        = (F1₁ + F1₂ + ... + F1ₙ) / n

👉 Simply the unweighted average across all classes

Example
Class A: 100 samples → F1 = 0.90
Class B: 10 samples  → F1 = 0.30

Macro F1 = (0.90 + 0.30) / 2 = 0.60

👉 Small class (B) is treated equally → important for imbalance

When to use Macro
✅ Use When
You care about ALL classes equally
Detecting performance on rare classes is important
Imbalanced dataset but fairness matters
Weighted Average

"Weight each class by how many samples it has."

Weighted Precision = (P₁×N₁ + P₂×N₂ + ... + Pₙ×Nₙ) / Total Samples
Weighted Recall    = (R₁×N₁ + R₂×N₂ + ... + Rₙ×Nₙ) / Total Samples
Weighted F1        = (F1₁×N₁ + F1₂×N₂ + ... + F1ₙ×Nₙ) / Total Samples

Where:

N₁, N₂, ... = number of samples in each class (support)
Example
Class A: 100 samples → F1 = 0.90
Class B: 10 samples  → F1 = 0.30

Weighted F1 = (100×0.90 + 10×0.30) / 110
            ≈ 0.85

👉 Large class dominates → more realistic overall performance

When to use Weighted
✅ Use When
Dataset is imbalanced
You want overall performance reflecting real data distribution
Majority class matters more in your problem
Micro Average

"Compute globally using total TP, FP, FN across all classes."

Micro Precision = Total TP / (Total TP + Total FP)
Micro Recall    = Total TP / (Total TP + Total FN)
Micro F1        = same as Micro Precision/Recall
Key Property
Micro Precision = Micro Recall = Micro F1

👉 Because everything is aggregated globally

When to use Micro
✅ Use When
You want a global performance metric
Classes are imbalanced
You don’t care about per-class performance
Code
from sklearn.metrics import precision_score, recall_score, f1_score

# Macro (equal weight per class)
precision_score(y_true, y_pred, average='macro')
recall_score(y_true, y_pred, average='macro')
f1_score(y_true, y_pred, average='macro')

# Weighted (weighted by support)
precision_score(y_true, y_pred, average='weighted')
recall_score(y_true, y_pred, average='weighted')
f1_score(y_true, y_pred, average='weighted')

# Micro (global)
precision_score(y_true, y_pred, average='micro')
recall_score(y_true, y_pred, average='micro')
f1_score(y_true, y_pred, average='micro')
Summary
Macro    → Treat all classes equally (fairness)
Weighted → Respect class imbalance (real-world distribution)
Micro    → Global counting (overall performance)

## 5. F-Beta Score

**Weighted balance — when you care more about P or R.**

```
F_β = (1 + β²) × (P × R) / (β² × P + R)

β < 1 → Precision weighted more
β = 1 → Equal (standard F1)
β > 1 → Recall weighted more

F0.5 → Precision 2× more important
F2   → Recall 2× more important
```

```python
from sklearn.metrics import fbeta_score

fbeta_score(y_actual, y_predicted, beta=2)    # recall 2x important (medical)
fbeta_score(y_actual, y_predicted, beta=0.5)  # precision 2x important (spam)
```

---

## 6. Specificity (True Negative Rate)

**"Of all actual negatives — how many correctly identified?"**

```
Specificity = TN / (TN + FP)

Our example: 6 / (6+1) = 0.857 = 85.7%
```

```
Recall      = how well you detect positives
Specificity = how well you detect negatives
Together    = full picture of model performance
```

---

## 7. ROC Curve and AUC

**Evaluates classifier across ALL thresholds simultaneously.**

```
ROC plots: TPR (Recall) vs FPR at every threshold

TPR = TP / (TP + FN)   = Recall
FPR = FP / (FP + TN)   = 1 - Specificity

ROC Curve shape:
TPR  |         ____________  ← Perfect model
1.0  |       /·
     |     /·   ← Your model
0.5  |   /·
     | /·········  ← Random (diagonal, AUC=0.5)
0.0  |/____________
     0.0    0.5    1.0   FPR
```

**AUC — Area Under Curve:**

```
AUC = 1.00  → Perfect
AUC > 0.90  → Excellent
AUC > 0.80  → Good
AUC > 0.70  → Fair
AUC = 0.50  → Random (useless)
AUC < 0.50  → Worse than random

Physical meaning:
AUC = probability that model ranks random positive
      higher than random negative
```

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_prob = pipeline.predict_proba(X_test)[:, 1]  # P(positive class)

fpr, tpr, thresholds = roc_curve(y_actual, y_prob)
auc = roc_auc_score(y_actual, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'gray', linestyle='--', label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 8. Precision-Recall Curve

**For heavily imbalanced datasets — ROC can be misleading.**

```
Why ROC misleads on imbalanced data:
995 healthy, 5 cancer → TN is huge
FPR = FP/(FP+TN) stays low even for bad models
→ AUC looks good even when rare class detection is poor

PR Curve focuses ONLY on positive class → honest for imbalanced data
```

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_actual, y_prob)
ap = average_precision_score(y_actual, y_prob)

plt.plot(recall, precision, label=f'PR Curve (AP={ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

---

## 9. Classification Report — All at Once

```python
from sklearn.metrics import classification_report

print(classification_report(y_actual, y_predicted,
      target_names=['Healthy', 'Cancer']))
```

```
              precision    recall  f1-score   support

     Healthy       0.86      0.86      0.86         7
      Cancer       0.67      0.67      0.67         3

    accuracy                           0.80        10
   macro avg       0.76      0.76      0.76        10
weighted avg       0.80      0.80      0.80        10

precision  → of predicted as this class: how many correct
recall     → of actual this class: how many caught
f1-score   → harmonic mean of both
support    → actual count of this class

macro avg     → simple average (ignores imbalance)
weighted avg  → weighted by class size (accounts for imbalance)
```

---

## Multiclass Confusion Matrix

```python
cm = confusion_matrix(y_actual, y_predicted,
                      labels=['Sports', 'Politics', 'Tech', 'Entertainment'])

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
       display_labels=['Sports', 'Politics', 'Tech', 'Entertainment'])
disp.plot(cmap='Blues')
plt.show()

# Reading it:
# Diagonal     = correct predictions ✅
# Off-diagonal = which class it confused with which ❌
```

---

## All Metrics — Cheat Sheet

```
Accuracy    = (TP+TN) / (TP+TN+FP+FN)
Precision   = TP / (TP+FP)               ← quality of positive predictions
Recall      = TP / (TP+FN)               ← coverage of actual positives
Specificity = TN / (TN+FP)               ← coverage of actual negatives
F1          = 2×P×R / (P+R)              ← harmonic mean
F_β         = (1+β²)×P×R / (β²P+R)      ← weighted F score
ROC-AUC     = area under TPR vs FPR      ← threshold-independent
PR-AUC      = area under P vs R          ← imbalanced datasets
```

---

## Decision Guide — Which Metric to Use

```
Balanced classes?
    └── Yes → Accuracy + F1 sufficient

Imbalanced classes?
    ├── FN costly? (cancer, fraud, COVID)
    │       → Recall + F2 score
    ├── FP costly? (spam, legal system)
    │       → Precision + F0.5
    ├── Need one balanced number?
    │       → F1 Score
    └── Need threshold-independent view?
            ├── Slightly imbalanced → ROC-AUC
            └── Heavily imbalanced → PR curve + Average Precision
```

---

## Common Mistakes

| Mistake | Why Wrong | Fix |
|---|---|---|
| Using accuracy on imbalanced data | Hides class-specific failures | Use F1, Recall, ROC-AUC |
| Optimizing threshold at 0.5 always | Business costs differ | Tune threshold based on FP vs FN cost |
| Using ROC-AUC on heavily imbalanced | TN dominates, misleads | Use PR curve instead |
| Reporting macro avg on imbalanced | Treats rare class equally | Report weighted avg + per-class metrics |
| Using predict instead of predict_proba for ROC | Need probabilities not labels | Always use predict_proba for curves |

---

## Quick Reference Code

```python
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, fbeta_score,
                              roc_auc_score, classification_report,
                              confusion_matrix, roc_curve,
                              precision_recall_curve)

# Core metrics
accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred)
recall_score(y_true, y_pred)
f1_score(y_true, y_pred)
fbeta_score(y_true, y_pred, beta=2)    # recall-heavy
fbeta_score(y_true, y_pred, beta=0.5)  # precision-heavy

# Probability-based
y_prob = pipeline.predict_proba(X_test)[:, 1]
roc_auc_score(y_true, y_prob)

# Full report
print(classification_report(y_true, y_pred))

# Confusion matrix values
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Custom threshold
y_pred_custom = (y_prob >= 0.3).astype(int)
```