import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_wine
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

wine = load_wine()
X = wine.data
Y = wine.target

x_train, x_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# ── DECISION TREE ────────────────────────────────
print("="*50)
print("TRAINING DECISION TREE")
print("="*50)

dt_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    min_samples_split=10,
    random_state=42
)
dt_model.fit(x_train, y_train)
dt_predictions = dt_model.predict(x_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print(f"Decision Tree Test Accuracy: {dt_accuracy:.2%}")

# ── RANDOM FOREST ────────────────────────────────
print("\n" + "="*50)
print("TRAINING RANDOM FOREST (100 TREES)")
print("="*50)

rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=3,
    min_samples_leaf=5,
    min_samples_split=10,
    random_state=42,
    max_features='sqrt',
    oob_score=True  # Add OOB score!
)
rf_model.fit(x_train, y_train)
rf_predictions = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Random Forest Test Accuracy: {rf_accuracy:.2%}")
print(f"OOB Score: {rf_model.oob_score_:.2%}")

# ── COMPARISON ───────────────────────────────────
print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"Decision Tree:  {dt_accuracy:.2%}")
print(f"Random Forest:  {rf_accuracy:.2%}")
print(f"Improvement:   +{(rf_accuracy-dt_accuracy)*100:.1f}%")
print("="*50)

# ── FEATURE IMPORTANCE COMPARISON ────────────────
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Decision Tree importance
dt_importance = pd.Series(
    dt_model.feature_importances_,
    index=wine.feature_names
).sort_values(ascending=False)

dt_importance.plot(kind='barh', ax=ax[0], color='steelblue')
ax[0].set_title('Decision Tree - Feature Importance')
ax[0].set_xlabel('Importance')
ax[0].invert_yaxis()

# Random Forest importance
rf_importance = pd.Series(
    rf_model.feature_importances_,
    index=wine.feature_names
).sort_values(ascending=False)

rf_importance.plot(kind='barh', ax=ax[1], color='forestgreen')
ax[1].set_title('Random Forest - Feature Importance')
ax[1].set_xlabel('Importance')
ax[1].invert_yaxis()

plt.tight_layout()
plt.savefig('dt_vs_rf_importance.png', dpi=300)
plt.show()

# ── CONFUSION MATRIX ─────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Decision Tree CM
cm_dt = confusion_matrix(y_test, dt_predictions)
# ConfusionMatrixDisplay(
#     confusion_matrix=cm_dt,
#     display_labels=wine.target_names
# ).plot(ax=ax[0], cmap='Blues')
# ax[0].set_title('Decision Tree')
ConfusionMatrixDisplay(confusion_matrix=cm_dt,display_labels=wine.target_names).plot(ax=ax[0],cmap='Blues')
ax[0].set_title('Decision Tree')

# Random Forest CM
cm_rf = confusion_matrix(y_test, rf_predictions)
ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=wine.target_names
).plot(ax=ax[1], cmap='Greens')
ax[1].set_title('Random Forest')

plt.tight_layout()
plt.savefig('dt_vs_rf_confusion.png', dpi=300)
plt.show()

# ── VISUALIZE ONE TREE FROM FOREST ───────────────
fig, ax = plt.subplots(figsize=(20, 10))
tree.plot_tree(
    rf_model.estimators_[0],
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    fontsize=10,
    filled=True,
    rounded=True
)
plt.title("One Tree from Random Forest (Tree #1 of 100)")
plt.tight_layout()
plt.savefig('one_tree_from_forest.png', dpi=300)
plt.show()

print("\nRandom Forest Feature Importance:")
print(rf_importance)