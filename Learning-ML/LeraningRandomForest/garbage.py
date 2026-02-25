import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.tree import DecisionTreeClassifier, plot_tree , BaseDecisionTree
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

wine = load_wine()

X = wine.data
Y = wine.target

df = pd.DataFrame(X, columns=wine.feature_names)
df['Outcome'] = Y
#print(df.head(100))
print(df.describe())
print(df.info())
print(df['Outcome'].value_counts())

x_train, x_test, y_train, y_test = train_test_split (
    X,
    Y,
    test_size=0.2,
    random_state = 42,
    stratify=Y
)


model = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth=3,
                                min_samples_leaf=5,
                                min_samples_split=10,
                                random_state=42,
                                max_features='sqrt'  # Default
                                )

model.fit(x_train,y_train)

# print("Best Score: ",round(model.best_score_,2))
test_predictions = model.predict(x_test)
train_predictions = model.predict(x_train)

print('Testing Report\n',classification_report(y_test,test_predictions))

print('\nTraining Report',classification_report(y_train,train_predictions))

print(f'Training Accuracy: {round(accuracy_score(y_train,train_predictions),2)}')

print(f'Test Accuracy: {round(accuracy_score(y_test,test_predictions),2)}')

fig, ax = plt.subplots(figsize = (12,6))

# tree.plot_tree(model,feature_names=wine.feature_names,class_names= wine.target_names,fontsize=10,filled=True,rounded=True)

# plt.tight_layout()
# plt.show()

cm = confusion_matrix(y_test, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wine.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()
print(cm)

importance = pd.Series(
    model.feature_importances_,
    index=wine.feature_names
)
sorted_importances = importance.sort_values(ascending=False)

print("\nRandom Forest Feature Importance:")
print(sorted_importances)

# Plot
plt.figure(figsize=(10, 6))
sorted_importances.plot(kind='barh', color='forestgreen')
plt.title('Feature Importance - Random Forest (100 Trees)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.gca().invert_yaxis()
plt.show()




"""

## Each Tree Gets Different Random Features!

```
You have 13 features total:
[alcohol, malic_acid, ash, ..., proline]

Tree 1 gets random 3-4 features:
[alcohol, ash, flavanoids, proline]

Tree 2 gets different random 3-4:
[malic_acid, color_intensity, hue, magnesium]

Tree 3 gets different random 3-4:
[alcohol, color_intensity, proline, ash]

...

Tree 100 gets different random 3-4:
[flavanoids, ash, magnesium, hue]
```

---

## How Many Features Per Tree?

```
Default: max_features='sqrt'

For 13 features:
√13 = 3.6 ≈ 3-4 features per tree

So each tree only sees
3-4 random features out of 13! ✅
```

---

## Why This Makes Random Forest Better

```
Single Decision Tree:
→ Sees ALL 13 features
→ Picks color_intensity (strongest)
→ Becomes dependent on it
→ If color_intensity wrong → tree wrong!

Random Forest:
Tree 1: No color_intensity → uses flavanoids
Tree 2: Has color_intensity → uses it
Tree 3: No flavanoids → uses proline
Tree 4: Different combo...
...
Tree 100: Another combo

Result:
→ Different trees strong in different areas
→ One tree's weakness = another's strength
→ Voting balances everything! 🔥
```

---

## The Complete Randomness

Each tree is random in TWO ways:

### 1. Random Data (Rows)
```
You have 142 training samples

Tree 1: Gets 142 samples WITH REPLACEMENT
        [sample_3, sample_7, sample_3, sample_12...]
        (some samples repeated, some missing)

Tree 2: Gets different 142 random samples
        [sample_1, sample_45, sample_1, sample_89...]
```

### 2. Random Features (Columns)
```
Tree 1: Gets features [A, C, F, J]
Tree 2: Gets features [B, D, F, K]
Tree 3: Gets features [A, B, E, L]
```

---

## Example Voting

```
New wine sample to predict:

Tree 1 (features A,C,F,J):  "Class 0"
Tree 2 (features B,D,F,K):  "Class 1"
Tree 3 (features A,B,E,L):  "Class 0"
Tree 4 (features C,E,G,M):  "Class 0"
...
Tree 100 (features ...):    "Class 0"

Count:
Class 0 → 73 votes
Class 1 → 21 votes
Class 2 → 6 votes

Winner: Class 0! ✅
```

---

## You Can Control This!

```python
# Default - uses √13 ≈ 3-4 features
model = RandomForestClassifier(
    max_features='sqrt'
)

# Use ALL 13 features (like Decision Tree)
model = RandomForestClassifier(
    max_features=None  # or max_features=13
)

# Use only 2 features per tree
model = RandomForestClassifier(
    max_features=2
)

# Use log2(13) ≈ 3.7 features
model = RandomForestClassifier(
    max_features='log2'
)
```

---

## Why Not Give All Features to All Trees?

```
If every tree sees ALL features:
→ All trees pick same strongest feature
→ All trees look similar
→ Less diversity
→ Voting doesn't help much!

With random features:
→ Each tree forced to find different patterns
→ More diversity
→ Better voting
→ Stronger forest! 🔥
```

---

## Summary

```
100 trees in your Random Forest:

Each tree trains on:
→ Random 142 samples (with replacement)
→ Random 3-4 features (out of 13)

Result:
→ 100 different perspectives
→ All vote on prediction
→ Majority wins
→ More accurate and stable! 💪
```

---



"""