import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib


def main():
    titanic_df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    print(titanic_df.head())
    print('Info\n')
    print(titanic_df.info())
    print('Description\n')
    print(titanic_df.describe())

    titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] 
    titanic_df = titanic_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Fare', 'Parch', 'SibSp'], axis=1)
    titanic_df.insert(len(titanic_df.columns) - 1, 'Survived', titanic_df.pop('Survived'))
    print(titanic_df.head())
    print('\nMissing Values\n')
    print(titanic_df.isnull().sum())
    titanic_df.fillna( {
        'Age': titanic_df['Age'].mean(),
        'Embarked': titanic_df['Embarked'].mode()[0],
    }, inplace= True)
    print('\n Missing Values After handling it ')
    print(titanic_df.isnull().sum())
    print()

    
    #handling Outliers

    #Ploting Outliers
    titanic_df['Sex'] = titanic_df['Sex'].astype('category')
    titanic_df['Embarked'] = titanic_df['Embarked'].astype('category')
    num_columns = titanic_df.drop('Survived',axis=1).select_dtypes(include=['number']).columns.tolist()
    cat_columns = titanic_df.select_dtypes(include=['category']).columns.tolist()


    print("---------- Bro -------------\n",num_columns)
    print("---------- Bro -------------\n",cat_columns)

    for column in num_columns:
        titanic_df[column]  = titanic_df[column].astype('float64')
    print(num_columns)
    fig, ax = plt.subplots(1,6,figsize=(12,6))

    for i , column in enumerate(num_columns):
        ax[i].boxplot(titanic_df[column])
        ax[i].set_title(column + "Box Plot")
    fig.suptitle("Checking Outliers using Box Plots(Before)")
    plt.tight_layout()
    plt.show()
    
    for column in num_columns:
        Q1 = titanic_df[column].quantile(0.25)
        Q3 = titanic_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        titanic_df[column] = titanic_df[column].clip(
            lower=lower, upper= upper,
        )

    fig, ax = plt.subplots(1,6,figsize=(12,6))

    for i , column in enumerate(num_columns):
        ax[i].boxplot(titanic_df[column])
        ax[i].set_title(column + "Box Plot")
    fig.suptitle("Checking Outliers using Box Plots(After)")
    plt.tight_layout()
    plt.show()

    # Splitting data

    X = titanic_df.drop(columns=['Survived'])
    Y = titanic_df['Survived']

    x_train, x_test, y_train, y_test = train_test_split(
        X,Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )


    # Encoding Categorical Features

    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 

    x_train_cat = x_train[cat_columns]

    x_train_num = x_train[num_columns]

    x_test_cat = x_test[cat_columns]

    x_test_num = x_test[num_columns]


    x_train_cat_enc = encoder.fit_transform(x_train_cat)
    
    x_test_cat_enc  = encoder.transform(x_test_cat)

    # Scalling Features
    scaler = StandardScaler()
    x_train_num_scl = scaler.fit_transform(x_train_num)
    x_test_num_scl = scaler.transform(x_test_num) 


    # NOW LET'S COMBINE THEM 
    x_train_final = np.hstack([x_train_cat_enc, x_train_num_scl])
    x_test_final = np.hstack([x_test_cat_enc, x_test_num_scl])



    # Choosing KNN Model

    # First let's find the best K

    """
    
    | Parameter       | Options                                 | Purpose                                                                                                   |
| --------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| n_neighbors (k) | int (3, 5, 7...)                        | Number of neighbors to consider. Most important! Small k = flexible/overfits. Large k = smooth/underfits. |
| weights         | 'uniform', 'distance'                   | 'uniform': All neighbors equal vote. 'distance': Closer neighbors vote more.                              |
| metric          | 'euclidean', 'manhattan', 'minkowski'   | Distance calculation method. Euclidean is default.                                                        |
| algorithm       | 'auto', 'ball_tree', 'kd_tree', 'brute' | Search method. 'auto' picks best.                                                                         |
| leaf_size       | int (30 default)                        | Tree-based algorithm tuning (speed vs accuracy).                                                          |
| p               | int (2 default)                         | Minkowski distance power (p=1 Manhattan, p=2 Euclidean).                                                  |
    """
    print('\nSetting KNN Model...')
    print('='*50,end='\n\n')
    best_k = -1
    best_score = -1
    for k in range(1,10):
        score = cross_val_score(
            KNeighborsClassifier(
             n_neighbors = k,
             weights = 'uniform',  # 'uniform': All neighbors equal vote. 'distance': Closer neighbors vote more.
             metric = 'euclidean',  # Distance calculation method. Euclidean is default.,
             algorithm = 'auto',
        ), x_train_final, y_train , cv=5, scoring='accuracy')
        print(f'K({k}) Score - > {score.mean():.2f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_k = k
    print(f'\nBest K Selected: {best_k}')
    print(f'\nBest Accuracy Score: {best_score:.2f}')
    knn_model = KNeighborsClassifier(n_neighbors=best_k, weights='uniform', metric='euclidean', algorithm='auto')
    knn_model.fit(x_train_final, y_train)
    print(f'Testing Model..')
    train_predictions = knn_model.predict(x_train_final)
    print(f'Training Accuracy Score: {accuracy_score(y_train, train_predictions):.2f} ')
    test_predictions = knn_model.predict(x_test_final)
    print(f'Testing Accuracy Score: {accuracy_score(y_test, test_predictions):.2f} ')
    knn_cm = confusion_matrix(y_test, test_predictions)
    print('KNN CM\n\n')
    print(knn_cm)
    print('='*50,end='\n\n')

    print('Selecting Logistic Regression Model....')
    print('+'*50,end='\n\n')
    lg_model = LogisticRegression(class_weight='balanced')
    lg_model.fit(x_train_final, y_train)
    lg_traing_predictions  = lg_model.predict(x_train_final)
    lg_test_predictions = lg_model.predict(x_test_final)
    lg_cm = confusion_matrix(y_test, lg_test_predictions)
    print(f'Logistic regressions Training Accurancy: {accuracy_score(y_train, lg_traing_predictions):.2f}')
    print(f'Logistic regressions Testing Accurancy: {accuracy_score(y_test, lg_test_predictions):.2f}')
    print("Logistic Regression Confusion Metric\n\n")
    print(lg_cm)
    print('+'*50,end='\n\n')

    print("Selecting Decision Tree Model....")
    print('+'*50,end='\n\n')
    dt_model = DecisionTreeClassifier(max_depth=3,criterion='gini',min_samples_split=10, min_samples_leaf=5, random_state=42)
    dt_model.fit(x_train_final, y_train)
    dt_train_pred = dt_model.predict(x_train_final)
    dt_test_pred = dt_model.predict(x_test_final)
    dt_cm = confusion_matrix(y_test, dt_test_pred)
    print(f'Decision Tree Training Accurancy: {accuracy_score(y_train, dt_train_pred):.2f}')
    print(f'Decision Tree Testing Accurancy: {accuracy_score(y_test, dt_test_pred):.2f}')
    print("Decision Tree Confusion Metric\n\n")
    print(dt_cm)
    print('+'*50,end='\n\n')

    print("Selecting randomforest  Model....")
    print('+'*50,end='\n\n')
    rf_model = RandomForestClassifier(n_estimators=100,criterion='gini',min_samples_split=10, min_samples_leaf=5, random_state=42)
    rf_model.fit(x_train_final, y_train)
    rf_train_pred = rf_model.predict(x_train_final)
    rf_test_pred = rf_model.predict(x_test_final)
    rf_cm = confusion_matrix(y_test, rf_test_pred)
    print(f'Random Forest Training Accurancy: {accuracy_score(y_train, rf_train_pred):.2f}')
    print(f'Random Forest Testing Accurancy: {accuracy_score(y_test, rf_test_pred):.2f}')
    print("Random Forest Confusion Metric\n\n")
    print(rf_cm)
    print('+'*50,end='\n\n')
    
    print('Saving All Models...')

    joblib.dump(lg_model, 'lg_titanic_model.joblib')
    joblib.dump(dt_model, 'dt_titanic_model.joblib')
    joblib.dump(rf_model, 'rf_titanic_model.joblib')
    joblib.dump(knn_model, 'knn_titanic_model.joblib')
    joblib.dump(encoder, 'titanic_encoder.joblib')
    joblib.dump(scaler, 'titanic_scaler.joblib')

    print('Models Saved!')




 

if __name__ == '__main__':
    main()