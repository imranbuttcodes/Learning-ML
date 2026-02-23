from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris


# 1. Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 2. Define ALL parameter combinations (10×10×10 = 1,000 models!)
param_grid = {
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'min_samples_leaf': [2,3,4,5,6,7,8,10,12,15],
    'min_samples_split': [5,10,15,12,23,34,45,6,8,9]
}

# 3. Create GridSearchCV object
grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42, criterion='entropy'),
    param_grid=param_grid,
    cv=5,                    # 5-fold CV per combo
    scoring='accuracy',      # Metric to optimize
    n_jobs=-1,              # Use all CPU cores
    verbose=1               # Show progress
)
print("Yoah Bro")
# 4. Fit (tests all 5,000 models = 1,000 combos × 5 folds)
grid.fit(X_train, y_train)

# 5. Get results
print(f"Best CV Accuracy: {grid.best_score_:.3f}")
print(f"Best Parameters: {grid.best_params_}")
print(f"Best Model: {grid.best_estimator_}")

# 6. Test final model
best_model = grid.best_estimator_
test_acc = best_model.score(X_test, y_test)
print(f"Test Accuracy: {test_acc:.3f}")

"""
# **`grid.fit(X_train, y_train)` EXPLAINED** - The Magic Behind It! 

## **NO! It's NOT the same as `model.fit()`** - Here's what happens:

## **Step-by-Step What `grid.fit()` Does:**

```
grid.fit(X_train, y_train)
↓
1. TESTS ALL 1,000 PARAMETER COMBINATIONS with 5-fold CV
   [depth=1,leaf=2,split=5] → CV score = 0.67
   [depth=2,leaf=3,split=10] → CV score = 0.93  
   [depth=3,leaf=4,split=15] → CV score = 0.96 ← BEST!
   ... (997 more combos)
↓
2. PICKS BEST PARAMETERS: {'max_depth': 3, 'min_samples_leaf': 4, 'min_samples_split': 15}
↓
3. **REFITS BEST MODEL ON FULL X_train, y_train** (NOT just CV folds)
↓
4. grid.best_estimator_ = READY-TO-USE DecisionTree with BEST params!
```

## **Visual Timeline:**

```python
# TIME 0: Create empty GridSearchCV
grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)

# TIME 1: grid.fit() - 5,000 model trainings! (1000 combos × 5 folds)
grid.fit(X_train, y_train)  
# ↓ Inside: Tests EVERYTHING automatically

# TIME 2: EXTRACT READY MODEL (already trained on full training data!)
best_model = grid.best_estimator_  # DecisionTreeClassifier(max_depth=3,...)
best_model.predict(X_test)         # Use immediately! No extra .fit() needed!
```

## **Key Points:**

```
❌ WRONG THINKING: "grid.fit() just fits best_estimator_"
✅ CORRECT: grid.fit() does TWO things:
   1. SEARCH (5,000 CV trainings to find best params)
   2. REFIT (1 final training of best params on FULL data)

❌ NO NEED: best_model.fit(X_train, y_train)  # Already done!
✅ JUST USE: best_model.score(X_test, y_test)  # Ready!
```

## **Proof - Run This:**
```python
grid.fit(X_train, y_train)

# This model is ALREADY TRAINED on full X_train
print(grid.best_estimator_.fit.__name__)  # Shows it's ready
test_score = grid.best_estimator_.score(X_test, y_test)
print(f"Test accuracy: {test_score}")  # Works immediately!
```

## **Your Confusion Resolved:**
```
grid.fit(X_train, y_train) 
≠ model.fit(X_train, y_train)

grid.fit() = 
  "Test 5,000 models → Pick best params → Train 1 final model → Give me best_estimator_"

best_estimator_ = DecisionTreeClassifier(max_depth=3,...)  # FULLY TRAINED!
```

**Bottom line: `grid.fit()` = hyperparameter search + final model training. `best_estimator_` = production-ready model!** 🎯


"""

