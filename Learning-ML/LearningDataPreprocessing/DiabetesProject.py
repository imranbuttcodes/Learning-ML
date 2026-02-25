import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score 

def HandleMissingValues(df,columns):
    for column in columns:
        median_val = df[column].median()
        mean_val = df[column].mean()
        df.loc[df[column] == 0 , column] = median_val if median_val != 0 else mean_val


def HandleOutliers(df,columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower=lower, upper=upper)


def PlotModelResults(cm, df, features, model, x_test, y_test):

    fig, ax = plt.subplots(1,3,figsize = (12,6))
    
    importance = abs(model.coef_[0])

    ax[0].bar(features, importance, color = 'green', edgecolor  = 'black')

    ax[0].set_xticks(range(len(features)))
    ax[0].set_xticklabels(features, ha = 'right', rotation = 45, fontsize = 9)

    ax[0].set_xlabel("Features", color = 'black', fontsize = 12, fontweight = 'bold')

    ax[0].set_ylabel("Feature Importance", color = 'black', fontsize = 12, fontweight = 'bold')

    ax[0].set_title('Feature Importance Bar', color = 'black', fontsize = 14, fontweight = 'bold')

    img = ax[1].imshow(cm, cmap = 'Blues')

    plt.colorbar(img, ax = ax[1])

    ax[1].set_title("Confusion Matrix",
                fontsize = 14,
                fontweight = 'bold')

    ax[1].set_xticks([0,1])
    ax[1].set_yticks([0,1])

    ax[1].set_xticklabels(['Non Diabetic', 'Diabetic'])
    ax[1].set_yticklabels(['Non Diabetic', 'Diabetic'])

    ax[1].set_xlabel('Predicted',fontsize = 14, fontweight = 'bold')
    ax[1].set_ylabel('Actual',fontsize = 14, fontweight = 'bold')

    for i in range(2):
        for j in range(2):
            ax[1].text (
                j,i,cm[i][j],
                color = 'black',
                fontsize = 12,
                va= 'center',
                ha = 'center'
            )


    probs = model.predict_proba(x_test)[:,1]
    ax[2].hist(probs[y_test==0], bins=20,
            alpha=0.6, color='blue',
            label='Not Diabetic')
    ax[2].hist(probs[y_test==1], bins=20,
            alpha=0.6, color='red',
            label='Diabetic')
    ax[2].axvline(x=0.5, color='black',
                linestyle='--',
                label='Decision Boundary')
    ax[2].set_title('Prediction Probability Distribution')
    ax[2].set_xlabel('Probability of Diabetic')
    ax[2].set_ylabel('Count')
    ax[2].legend()        
    fig.suptitle('Diabetes Prediction System — Model Results',
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig( "04_model_results.png",dpi = 300)
    plt.show()


def NormalizeFeatures(df,columns):
    scalor = StandardScaler()
    df[columns] = scalor.fit_transform(df[columns])
    return scalor


def plotBox(df, title, columns, file_name):
    fig ,ax = plt.subplots(1,7,figsize= (20,8))

    boxplots_titles = np.array(['Outliers in Glucose' , 'Outliers in BloodPressure' , 'Outliers in SkinThickness',  'Outliers in Insulin' ,  'Outliers in BMI',  "Outliers in DiabetesPedigreeFunction" , 'Outliers in Age' ])
    fig.suptitle(title, fontsize = 14, fontweight = 'bold')
    for i , column in enumerate(columns):
        ax[i].boxplot(df[column])
        ax[i].set_title(boxplots_titles[i],fontweight = 'bold', fontsize = 10)
    
    plt.tight_layout()
    fig.savefig(file_name,dpi=300)
    plt.show()

def FeatureEngeering(df):
    df.drop(columns=['Insulin','SkinThickness', 'BloodPressure'],inplace = True)
    df['HighGlucose'] = np.where(
        df['Glucose'] > 140, 1, 0
    )
    df['HighBMI'] = np.where(
        df['BMI'] > 30, 1, 0
    )

    df['BMI_Glucose'] = df['BMI'] * df['Glucose']
    df['Age_Glucose'] = df['Age'] * df['Glucose']


def InitialInsights(diabetes_df):
    print("Shape", diabetes_df.shape)
    print("\nFirst 20 Entities\n")
    print(diabetes_df.head(20).to_string())
    print(diabetes_df.info())
    print(diabetes_df.describe(),end='\n\n')
    columns = ['Glucose' , 'BloodPressure' , 'SkinThickness',  'Insulin' ,  'BMI',  "DiabetesPedigreeFunction" , 'Age' ]
    for column in columns:
        diabetes_df[column] = diabetes_df[column].astype('float64')
    
    print("Missing Values")

    for column in columns:
        zero_counts = (diabetes_df[column] == 0).sum()
        print(column)
        print(f"Zero Counts -> {zero_counts}",end='\n\n')    
    
    print(diabetes_df['Outcome'].value_counts())
    

    fig_1, ax_1 = plt.subplots(1,3,figsize = (18,6))
    class_counts = diabetes_df['Outcome'].value_counts()
    ax_1[0].bar(['Non Diabetic', 'Diabetic'],class_counts, color = ['#2563EB', '#DC2626'], edgecolor = 'black')
    ax_1[0].set_title('Class Distribution',fontweight='bold', fontsize=12)
    fig_1.suptitle("Data Overview")
    ax_1[0].set_xlabel('Class', fontweight='bold')
    ax_1[0].set_ylabel('Count', fontweight='bold')
    for i, val in enumerate(class_counts.values):
        ax_1[0].text(i, val + 5, str(val),
                    ha='center', fontweight='bold')


    ax_1[1].hist(diabetes_df['Glucose'], bins=20,
                color='#2563EB', edgecolor='black')
    ax_1[1].set_title('Glucose Distribution',
                    fontweight='bold', fontsize=12)
    ax_1[1].set_xlabel('Glucose', fontweight='bold')
    ax_1[1].set_ylabel('Count', fontweight='bold')
    ax_1[1].axvline(diabetes_df['Glucose'].mean(),
                color='red', linestyle='--',
                linewidth=2, label='Mean')
    ax_1[1].legend()

    diabetic     = diabetes_df[diabetes_df['Outcome'] == 1]
    non_diabetic = diabetes_df[diabetes_df['Outcome'] == 0]

    ax_1[2].scatter(non_diabetic['Glucose'],
                non_diabetic['BMI'],
                color='#2563EB', edgecolor='black',
                s=30, alpha=0.6, label='Not Diabetic')
    ax_1[2].scatter(diabetic['Glucose'],
                diabetic['BMI'],
                color='#DC2626', edgecolor='black',
                s=30, alpha=0.6, label='Diabetic')
    ax_1[2].set_title('BMI vs Glucose',
                    fontweight='bold', fontsize=12)
    ax_1[2].set_xlabel('Glucose', fontweight='bold')
    ax_1[2].set_ylabel('BMI', fontweight='bold')
    ax_1[2].legend()

    plt.tight_layout()
    plt.savefig('01_data_overview.png', dpi=300)
    plt.show()

def main():
    diabetes_df = pd.read_csv('diabetes.csv')

    InitialInsights(diabetes_df)
    columns = ['Glucose' , 'BloodPressure' , 'SkinThickness',  'Insulin' ,  'BMI',  "DiabetesPedigreeFunction" , 'Age' ]
    print('-------------------- Before Hanlding Missing Values --------------------')
    print(diabetes_df,end='\n\n')

    HandleMissingValues(diabetes_df,columns)

    print('-------------------- After Hanlding Missing Values --------------------')
    print(diabetes_df,end='\n\n')

    plotBox(diabetes_df,"Before Handling Outliers", columns,'02_outliers_before.png')

    HandleOutliers(diabetes_df,columns)

    plotBox(diabetes_df,"After Handling Outliers", columns,'03_outliers_after.png')

    print('Applying Feature Engineering.....\n')
    FeatureEngeering(diabetes_df)
    print("---------------------------- Dataset After Applying Feature Engineering----------------------------\n")
    print(diabetes_df,end='\n\n')

    print("Scalling Features........")

    scale = NormalizeFeatures(diabetes_df, diabetes_df.drop('Outcome', axis=1).columns)
    
    print("---------------------------- Dataset After Applying Scalling Features----------------------------\n")

    print(diabetes_df)

    X = diabetes_df.drop('Outcome',axis = 1).values

    Y = np.array(diabetes_df['Outcome'])
    model = LogisticRegression(class_weight='balanced')

    x_train, x_test, y_train, y_test = train_test_split (
        X, Y,
        test_size = 0.2,
        random_state = 42,
        stratify = Y
    )

    print("Training Model....")

    model.fit(x_train, y_train)

    print("Model Trained")

    print("------------------ Evalauation Matrices ------------------")
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1Score = f1_score(y_test, predictions)

    cm = confusion_matrix(y_test, predictions)

    evaluation_metric = f'Accuracy = {accuracy * 100 :.2f}%\nPrecision = {precision * 100 :.2f}%\nRecall = {recall * 100:.2f}%\nF1 Score = {f1Score:.2f}'

    print(evaluation_metric)

    print("Confusion Metic")

    print(cm)

    PlotModelResults(cm,diabetes_df,diabetes_df.drop('Outcome',axis = 1).columns.tolist(),model,x_test,y_test)

    print("---------------------- Prediction Results ----------------------")
    
    print("Total testing sample", len(x_test))

    for i, val in enumerate(x_test):
        print(f'x = {val}, actual = {y_test[i]}, predicted = {predictions[i]}\n')


if __name__ == '__main__':
    main()
