import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Titanic data
titanic_df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Feature Engineering (you know this!)
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch']
titanic_df = titanic_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 
                               'Fare', 'Parch', 'SibSp'], axis=1)

# Handle missing values
titanic_df = titanic_df.fillna({
    'Age': titanic_df['Age'].mean(),
    'Embarked': titanic_df['Embarked'].mode()[0]
})

# Separate features
X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Identify columns
num_columns = ['Pclass', 'Age', 'FamilySize']
cat_columns = ['Sex', 'Embarked']

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

print("="*50)
print("SVM ON TITANIC DATASET")
print("="*50)

# Try different kernels
kernels = ['linear', 'rbf']

for kernel in kernels:
    print(f"\n{kernel.upper()} SVM:")
    
    svm = SVC(kernel=kernel, C=1.0, random_state=42)
    svm.fit(X_train_final, y_train)
    
    train_acc = svm.score(X_train_final, y_train)
    test_acc = svm.score(X_test_final, y_test)
    
    print(f"  Training: {train_acc:.2%}")
    print(f"  Testing:  {test_acc:.2%}")

# GridSearch for best SVM
print("\n" + "="*50)
print("FINDING BEST SVM")
print("="*50)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, verbose=1)
grid.fit(X_train_final, y_train)

print(f"\nBest Parameters: {grid.best_params_}")
print(f"Best CV Score: {grid.best_score_:.3f}")

best_svm = grid.best_estimator_
test_pred = best_svm.predict(X_test_final)

print(f"\nFinal Test Accuracy: {accuracy_score(y_test, test_pred):.2%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))