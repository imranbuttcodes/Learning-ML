import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.tree import DecisionTreeClassifier, plot_tree , BaseDecisionTree
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

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

grid_param = {
    'max_depth' : [3,4,5],
    'min_samples_leaf' : [5,10,15],
    'min_samples_split': [10,15,23],
}


# it will automaticall do Hyperparameter Tuning 
grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=grid_param,
    n_jobs=-1,                              
    scoring='accuracy',
    verbose=1,
    cv=5
)
grid.fit(x_train,y_train)
print(grid.best_params_)
print("Best Score: ",round(grid.best_score_,2))
model = grid.best_estimator_
test_predictions = model.predict(x_test)
train_predictions = model.predict(x_train)

print('Testing Report\n',classification_report(y_test,test_predictions))

print('\nTraining Report',classification_report(y_train,train_predictions))

print(f'Training Accuracy: {round(accuracy_score(y_train,train_predictions),2)}')

print(f'Test Accuracy: {round(accuracy_score(y_test,test_predictions),2)}')

fig, ax = plt.subplots(figsize = (12,6))

tree.plot_tree(model,feature_names=wine.feature_names,class_names= wine.target_names,fontsize=10,filled=True,rounded=True)

plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wine.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()
print(cm)
importance = pd.Series(model.feature_importances_,index=wine.feature_names)
sorted_importances = importance.sort_values(ascending=False)
print(sorted_importances)
plt.figure(figsize=(10, 6))
sorted_importances.plot(kind='barh', color='teal')
plt.title('Feature Importance in Wine Classification (Decision Tree)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.gca().invert_yaxis()  # Put the most important at the top
plt.show()