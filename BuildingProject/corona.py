import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,  MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score
from sklearn.impute import SimpleImputer
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
    #fig, ax = plt.subplots(1,2,figsize= (12,6))
    class_count = df['covid_result'].value_counts()
    print('\n',class_count)
    #plt.bar(['positive results', 'negative results'], class_count, edgecolor = 'black', color = ["#690716", "#1F7D03"])
    #plt.show()

    cat_col = df.select_dtypes(include=['category', 'object']).columns.tolist()
  

    num_col = df.drop('covid_result', axis=1).select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print('Categorical Features\n')
    print(cat_col)
   
    print('Numerical Features\n')
    print(num_col)

    # Handle missing values

    
    df['comorbidity'] = df['comorbidity'].fillna('None')
    print(df.isnull().sum())
    
    # Split data

    X = df.drop('covid_result',axis=1)
    Y = df['covid_result']
    

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42,
        stratify=Y,
    )
    # Encoding Categorical  data
    x_train = pd.get_dummies(x_train, columns=['comorbidity'], prefix='comorbidity')
    print(x_train.columns.to_list())
    x_test = pd.get_dummies(x_test, columns=['comorbidity'],prefix='comorbidity')

    print('\n\n',x_test.columns.to_list())
if __name__ == '__main__':
    main()    




