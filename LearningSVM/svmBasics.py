import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load Iris dataset (simple classification)
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# Use only 2 classes for visualization
X = X[y != 2]  # Remove class 2
y = y[y != 2]  # Now only 0 and 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# IMPORTANT: SVM needs scaling! ✅
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# BUILD LINEAR SVM
print("="*50)
print("LINEAR SVM")
print("="*50)

svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)

# Predictions
train_pred = svm_linear.predict(X_train_scaled)
test_pred = svm_linear.predict(X_test_scaled)

# Evaluate
print(f"\nTraining Accuracy: {accuracy_score(y_train, train_pred):.2%}")
print(f"Testing Accuracy:  {accuracy_score(y_test, test_pred):.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(y_test, test_pred))


# BUILD RBF SVM
print("\n" + "="*50)
print("RBF SVM")
print("="*50)

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)

# Predictions
train_pred = svm_rbf.predict(X_train_scaled)
test_pred = svm_rbf.predict(X_test_scaled)

# Evaluate
print(f"\nTraining Accuracy: {accuracy_score(y_train, train_pred):.2%}")
print(f"Testing Accuracy:  {accuracy_score(y_test, test_pred):.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

