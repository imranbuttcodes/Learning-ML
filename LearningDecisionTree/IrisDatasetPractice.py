import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score, precision_score



def H_para_tuning(X,Y):
    m_depth = [1,2,3,4,5,6,7,8,9,10]
    min_s_l = [2,3,4,5,6,7,8,10,12,15]
    min_s_s = [5,10,15,12,23,34,45,6,8,9]
    best_parameters = []
    best_score = 0
    for i in range(10):
        score = cross_val_score(DecisionTreeClassifier(
            max_depth=m_depth[i],
            min_samples_leaf=min_s_l[i],
            min_samples_split=min_s_s[i],
            random_state=42,
        ), X, Y,cv=5)
        print(f"Params {i}: {score.mean():.3f}")  # Debug
        if score.mean() > best_score:
            best_score = score.mean()
            best_parameters = [m_depth[i],min_s_l[i], min_s_s[i]]

    return tuple(best_parameters)


def industry_style_tuning(x_train, y_train):
    # But it cose more computation power
    param_grid = {
        'max_depth': [3,5,7],      
        'min_samples_split': [10,20],
        'min_samples_leaf': [5,10]
    }
    
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,           # Parallel
        scoring='accuracy',
        verbose=1            # Progress
    )
    return grid.fit(x_train, y_train)


def main():
    iris = load_iris()
    X = iris.data
    Y = iris.target

    #Splitting Data

    x_train, x_test , y_train, y_test = train_test_split(
        X,Y,test_size=0.2,random_state=42,
        stratify=Y
    )
    m_depth, min_s_l, min_s_s = H_para_tuning(x_train,y_train)
    print("Best Parameters\n")
    print(m_depth,min_s_l, min_s_s)
    model = DecisionTreeClassifier(
        max_depth=m_depth,
        min_samples_leaf = min_s_l,
        min_samples_split = min_s_s,
        random_state=42,
        criterion = 'entropy'
    )
    model.fit(x_train,y_train)
    train_predictions = model.predict(x_train)
    test_prediction = model.predict(x_test)
    print(f'Train Accuracy: {accuracy_score(y_train,train_predictions):.2f}')
    print(f"Testing Accuracy: {accuracy_score(y_test, test_prediction):.2f}")
    print(f'Train Recall {recall_score(y_train,train_predictions,average='macro'):.2f}')
    print(f'Test Recall {recall_score(y_test,test_prediction,average='macro'):.2f}')
    print(f'Train Precision {precision_score(y_train,train_predictions, average='macro'):.2f}')
    print(f'Test Precision {precision_score(y_test,test_prediction, average='macro'):.2f}')
   # print("Training Classification Report:\n",classification_report(y_train, train_predictions, target_names=iris.target_names))
   # print("Testing Classification Report:\n",classification_report(y_test, test_prediction, target_names=iris.target_names))

if __name__ == '__main__':
    main()






