import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load simple dataset
iris = load_iris()
X = iris.data
y = iris.target

# Use only 2 classes for simplicity
X = X[y != 2]
y = y[y != 2]

print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
print(f"Classes: 0 and 1")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# IMPORTANT: Always scale for SVM!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*50)
print("STEP 1: LINEAR SVM")
print("="*50)

# Build LINEAR SVM
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

# Predictions
y_pred_linear = svm_linear.predict(X_test_scaled)

# Evaluate
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"\nLinear SVM Accuracy: {accuracy_linear:.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_linear))

print("\n" + "="*50)
print("STEP 2: RBF SVM")
print("="*50)

# Build RBF SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

# Predictions
y_pred_rbf = svm_rbf.predict(X_test_scaled)

# Evaluate
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"\nRBF SVM Accuracy: {accuracy_rbf:.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rbf))

print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"Linear SVM: {accuracy_linear:.2%}")
print(f"RBF SVM:    {accuracy_rbf:.2%}")

if accuracy_rbf > accuracy_linear:
    print("\n🏆 RBF SVM wins!")
else:
    print("\n🏆 Linear SVM wins!")


    print("\n" + "="*50)
print("TESTING DIFFERENT C VALUES")
print("="*50)

C_values = [0.1, 1, 10, 100]

for C in C_values:
    svm = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    train_acc = svm.score(X_train_scaled, y_train)
    test_acc = svm.score(X_test_scaled, y_test)
    
    print(f"\nC = {C}:")
    print(f"  Train: {train_acc:.2%}")
    print(f"  Test:  {test_acc:.2%}")
    
    # Check overfitting
    gap = train_acc - test_acc
    if gap > 0.1:
        print(f"  ⚠️ Overfitting! Gap = {gap:.2%}")
    else:
        print(f"  ✅ Good!")

        print("\n" + "="*50)
print("TESTING DIFFERENT GAMMA VALUES")
print("="*50)

gamma_values = [0.001, 0.01, 0.1, 1, 10]

for gamma in gamma_values:
    svm = SVC(kernel='rbf', C=1.0, gamma=gamma, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    train_acc = svm.score(X_train_scaled, y_train)
    test_acc = svm.score(X_test_scaled, y_test)
    
    print(f"\ngamma = {gamma}:")
    print(f"  Train: {train_acc:.2%}")
    print(f"  Test:  {test_acc:.2%}")
    
    gap = train_acc - test_acc
    if gap > 0.15:
        print(f"  ⚠️ Overfitting badly! Gap = {gap:.2%}")
    elif gap > 0.05:
        print(f"  ⚠️ Slight overfitting. Gap = {gap:.2%}")
    else:
        print(f"  ✅ Good generalization!")

        from sklearn.model_selection import GridSearchCV

print("\n" + "="*50)
print("FINDING BEST PARAMETERS (GridSearch)")
print("="*50)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

# GridSearch
grid = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    verbose=1
)

print("Searching... (this takes a minute)")
grid.fit(X_train_scaled, y_train)

print(f"\n✅ Best Parameters: {grid.best_params_}")
print(f"✅ Best CV Score: {grid.best_score_:.3f}")

# Test best model
best_svm = grid.best_estimator_
test_score = best_svm.score(X_test_scaled, y_test)
print(f"✅ Test Accuracy: {test_score:.3f}")