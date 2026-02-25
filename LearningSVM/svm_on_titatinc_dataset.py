import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

print("="*60)
print("SVM ON TITANIC DATASET")
print("="*60)

# Load Titanic
titanic_df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
print(titanic_df)
# Feature Engineering
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch']
titanic_df = titanic_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 
                               'Fare', 'Parch', 'SibSp'], axis=1)

# Handle missing values
titanic_df = titanic_df.fillna({
    'Age': titanic_df['Age'].mean(),
    'Embarked': titanic_df['Embarked'].mode()[0]
})

# Features and target
X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Identify column types
num_columns = ['Pclass', 'Age', 'FamilySize']
cat_columns = ['Sex', 'Embarked']

# OR

# num_columns = titanic_df.select_dtypes(include = ['int64', 'float64']).columns.tolist()
# cat_columns = titanic_df.select_dtypes(include = ['category', 'object']).columns.tolist()

# Encode categorical
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train[cat_columns])
X_test_cat = encoder.transform(X_test[cat_columns])

# Scale numerical (CRITICAL FOR SVM!)
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_columns])
X_test_num = scaler.transform(X_test[num_columns])

# Combine
X_train_final = np.hstack([X_train_cat, X_train_num])
X_test_final = np.hstack([X_test_cat, X_test_num])

# Try Linear SVM
print("\nLINEAR SVM:")
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_final, y_train)

train_acc_linear = svm_linear.score(X_train_final, y_train)
test_acc_linear = svm_linear.score(X_test_final, y_test)

print(f"  Training: {train_acc_linear:.2%}")
print(f"  Testing:  {test_acc_linear:.2%}")

# Try RBF SVM
print("\nRBF SVM:")
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_final, y_train)

train_acc_rbf = svm_rbf.score(X_train_final, y_train)
test_acc_rbf = svm_rbf.score(X_test_final, y_test)

print(f"  Training: {train_acc_rbf:.2%}")
print(f"  Testing:  {test_acc_rbf:.2%}")

# GridSearch for best SVM
print("\n" + "="*60)
print("FINDING BEST SVM WITH GRIDSEARCH")
print("="*60)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, verbose=1)
grid.fit(X_train_final, y_train)

print(f"\nBest Parameters: {grid.best_params_}")
print(f"Best CV Score: {grid.best_score_:.3f}")

# Final evaluation
best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test_final)

print(f"\nFinal Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

print("\n" + "="*60)
print("COMPARING ALL 7 ALGORITHMS")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes': GaussianNB(),
    'SVM Linear': SVC(kernel='linear', C=1.0, random_state=42),
    'SVM RBF': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
}

results = []

for name, model in models.items():
    # Train
    model.fit(X_train_final, y_train)
    
    # Scores
    train_acc = model.score(X_train_final, y_train)
    test_acc = model.score(X_test_final, y_test)
    
    results.append({
        'Model': name,
        'Train': f"{train_acc:.2%}",
        'Test': f"{test_acc:.2%}",
        'Overfit': f"{(train_acc - test_acc):.2%}"
    })
    
    print(f"\n{name}:")
    print(f"  Train: {train_acc:.2%}")
    print(f"  Test:  {test_acc:.2%}")

# Summary table
print("\n" + "="*60)
print("FINAL COMPARISON TABLE")
print("="*60)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Winner
best_idx = results_df['Test'].str.rstrip('%').astype(float).idxmax()
print("\n" + "="*60)
print(f"🏆 WINNER: {results_df.iloc[best_idx]['Model']}")
print(f"   Test Accuracy: {results_df.iloc[best_idx]['Test']}")
print("="*60)

"""

**OvR trains K separate binary models. Multinomial trains 1 joint model.** Different decision boundaries, probabilities, and performance.

## Key Differences

| Aspect | One-vs-Rest (OvR) | Multinomial (Softmax) |
|--------|-------------------|----------------------|
| **Models Trained** | K binary classifiers | 1 model for all classes |
| **Probability Constraint** | Independent (may not sum to 1) | Sum to 1 (P1+P2+P3=1) |
| **Decision Boundaries** | Different per class | Joint optimization |
| **Scikit-learn** | `LogisticRegression()` (default) | `multi_class='multinomial'` |
| **Best For** | Imbalanced classes | Mutually exclusive classes |

## Code Example
```python
# OvR (3 models for Iris)
lr_ovr = LogisticRegression()  # Class0-vs-rest, Class1-vs-rest, Class2-vs-rest

# Multinomial (1 model)
lr_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
```

## When They Differ Most
```
Dataset: 3 overlapping classes
OvR:     May produce jagged boundaries (class1 boundary different from others)
Multi:   Smooth, consistent boundaries across all classes
```

## Your Choice
```
✅ Titanic (2 classes): Both identical
✅ Iris (3 classes): Use Multinomial - better calibrated probabilities
✅ Production: Multinomial recommended [web:67]
```

**Multinomial > OvR for most cases** - joint optimization gives better generalization! 🚀

"""