import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib as jb
def load_data():
    return pd.read_csv("covid19_patient_symptoms_diagnosis.csv")


def main():
    df = load_data()
    print(df.head())
    print(df.describe())
    print('Missing Values\n')
    print(df.isnull().sum())
    print(df['covid_result'].value_counts())
    print('Missing Values Percentage %\n')
    print(df.isnull().sum() / len(df) * 100)
    fig, ax = plt.subplots(1,3,figsize= (18,5))
    class_count = df['covid_result'].value_counts()
    print('\n',class_count)
    ax[0].bar(['positive results', 'negative results'], class_count, edgecolor = 'black', color = ["#690716", "#1F7D03"])
    ax[0].set_title('COVID Class Distribution')
    cat_col = df.select_dtypes(include=['category', 'object']).columns.tolist()
  

    num_col = df.drop('covid_result', axis=1).select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print('Categorical Features\n')
    print(cat_col)
   
    print('Numerical Features\n')
    print(num_col)

    # Handle missing values

    
    df['comorbidity'] = df['comorbidity'].fillna('No_Comorbidity')
    print(df.isnull().sum())
    
    # Split data
    #X = df.drop(['covid_result', 'patient_id'],axis=1)

    X = df.drop(['covid_result', 'patient_id',
            'dry_cough', 'fever'], axis=1)


    Y = df['covid_result']
    

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42,
        stratify=Y,
    )
    # Encoding Categorical  data
    #One Hot Encoding 
    x_train = pd.get_dummies(x_train, columns=['comorbidity'], prefix='comorbidity')
    x_test = pd.get_dummies(x_test, columns=['comorbidity'],prefix='comorbidity')
    
    x_train = pd.get_dummies(x_train,columns=['gender'],prefix='gender')
    x_test = pd.get_dummies(x_test,columns=['gender'],prefix='gender')

    #Alignment     
    x_train, x_test = x_train.align(x_test,join='left', axis=1, fill_value=0)

    print("Training Data Insights\n\n",x_train.describe())
    print("Testing Data Insights\n\n",x_test.describe())
        
    print(f'{"="*10} LG Model {"="*10}\n')
        
    numeric_features = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), numeric_features),
        ('passthrough', 'passthrough', x_train.columns.difference(numeric_features))
    ])

    x_train_scaled = preprocessor.fit_transform(x_train)
    x_test_scaled = preprocessor.transform(x_test)
    print('Saving Preprocessor......')    
    jb.dump(preprocessor,'preprocessor.joblib')
    x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=preprocessor.get_feature_names_out(), index=x_train.index)
    x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=preprocessor.get_feature_names_out(), index=x_test.index)
    
    lg_model = LogisticRegression(class_weight={0:1,1:2}, random_state=42,
                                  )
    lg_model.fit(x_train_scaled_df, y_train)
    lg_test_predictions = lg_model.predict(x_test_scaled_df)
    lg_train_predictions = lg_model.predict(x_train_scaled_df)
   
    print(f'Training Accuracy = {accuracy_score(y_train,lg_train_predictions):.2f}') 
    print(f'Testing Accuracy = {accuracy_score(y_test,lg_test_predictions):.2f}')
    print(f'Recall: {recall_score(y_test,lg_test_predictions):.2f}')
    print(f'Precision Score: {precision_score(y_test, lg_test_predictions):.2f}')
    lg_cm = confusion_matrix(y_test, lg_test_predictions)
    print('----------------- Confusion Metrix -----------------\n')
    print(lg_cm,end='\n\n')
    print('Saving Logistic Regression Model......')    
    jb.dump(lg_model,'corona_lg_model.joblib')
    print(f'{"="*10}==========={"="*10}\n')


    # print(f'{"="*10} Random Forest {"="*10}\n')
    # rf_param_grid = {
    # 'max_depth': [3, 5, 7, 10],
    # 'min_samples_split': [5, 10, 20],
    # 'min_samples_leaf': [2, 5, 10],
    # 'criterion': ['gini', 'entropy']
    # }


    # rf_model = RandomForestClassifier(n_estimators=100,criterion='gini',)
    # grid_search = GridSearchCV(rf_model,rf_param_grid, scoring='accuracy',n_jobs=-1)
    # grid_search.fit(x_train_scaled_df,y_train)
    # rf_model = grid_search.best_estimator_
    


    # rf_test_predictions = rf_model.predict(x_test_scaled_df)
    # rf_train_predictions = rf_model.predict(x_train_scaled_df)

    # print(f'Training Accuracy = {accuracy_score(y_train,rf_train_predictions):.2f}') 
    # print(f'Testing Accuracy = {accuracy_score(y_test,rf_test_predictions):.2f}')
    # print(f'Recall: {recall_score(y_test,rf_test_predictions):.2f}')
    # print(f'Precision Score: {precision_score(y_test, rf_test_predictions):.2f}')
    # rf_cm = confusion_matrix(y_test, lg_test_predictions)
    # print('----------------- Confusion Metrix -----------------\n')
    # print(rf_cm,end='\n\n')

    # print('Saving RandomForest model......')    
    # jb.dump(rf_model,'corona_rf_model.joblib')

    # print(f'{"="*10}==========={"="*10}\n')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=lg_cm, display_labels=['No Covid', 'Covid'])

    disp.plot(cmap='Blues',ax=ax[1])

    ax[1].set_title('Logistic Regression COvid Model Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()    