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

def load_data():
    return pd.read_csv("covid19_patient_symptoms_diagnosis.csv")


def main():
    df = load_data()
    print(df.head())
    print(df.describe())

    print(df['comorbidity'].isnull().sum())
    print(df['covid_result'].value_counts())

    fig, ax = plt.subplots(1,2,figsize= (12,6))
    class_count = df['covid_result'].value_counts()
    plt.bar(['positive results', 'negative results'], class_count, edgecolor = 'black', color = ["#690716", "#1F7D03"])
    plt.show()

if __name__ == '__main__':
    main()    



