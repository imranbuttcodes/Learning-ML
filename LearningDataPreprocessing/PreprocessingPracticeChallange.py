import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score 

data = {
    "StudentID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20],

    "Age":       [18, 19, np.nan, 17, 20, 18, 999, 19, 18, 20,
                  17, 18, np.nan, 19, 20, 18, 17, np.nan, 19, 18],

    "Gender":    ["Male", "Female", "Male", "Female", "Male",
                  "Female", "Male", "Female", np.nan, "Male",
                  "Female", "Male", "Female", "Male", np.nan,
                  "Female", "Male", "Female", "Male", "Female"],

    "StudyHours":[5, 8, 3, 9, 6, 7, 2, 8, 5, 9,
                  4, 6, 7, 8, 3, 5, 9, 6, 7, 4],

    "Attendance":[85, 92, 78, 95, 88, 91, 60, 94, 82, 97,
                  75, 87, 90, 93, 70, 84, 96, 89, 91, 80],

    "MidTerm":   [65, 82, 55, 88, 72, 79, 45, 85, 68, 91,
                  58, 75, 80, 84, 52, 69, 90, 77, 81, 63],

    "Assignment":[70, 85, 60, 90, 75, 82, 50, 88, 72, 93,
                  62, 78, 83, 87, 58, 71, 92, 80, 84, 67],

    "Grade":     ["C", "A", "D", "A", "B", "B", "F", "A", "C", "A",
                  "D", "B", "B", "A", "F", "C", "A", "B", "B", "C"],

    "Passed":    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
                  0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Shape of the dataset",df.shape)
print("\n---------------------- Initial Messy Dataset ----------------------\n")
print(df.to_string())

print('\nMissing Values\n')

print(df.isnull().sum())
print('\nMissing Value Percentage%\n')
print(df.isnull().sum() / len(df) * 100)

fig, ax = plt.subplots(1,4,figsize = (18, 6))

#Plotting Histogram of studyHours
ax[0].hist(df['StudyHours'],bins = 5,edgecolor = 'black')

ax[0].set_title('Histogram of Students Study Hours',fontsize = 14, fontweight = 'bold')

ax[0].set_xlabel('Study Hours', fontsize = 12, fontweight = 'bold')

ax[0].set_ylabel('Total Students', fontsize = 12, fontweight = 'bold')

ax[1].scatter(df['StudyHours'], df['MidTerm'],s=50,edgecolor = 'black',label = 'Mid term Marks\nBased on StudyHours', color = 'green')

ax[1].set_title('Study Hours Vs MidTerm', color = 'black', fontsize = 14, fontweight = 'bold')

ax[1].set_xlabel('Study Hours', color = 'black', fontsize = 12, fontweight = 'bold')

ax[1].set_ylabel('MidTerm Marks', color = 'black', fontsize = 12, fontweight = 'bold')

ax[1].set_ylim(0,100)

ax[1].set_yticks(np.arange(0,100,5))



Q1 = df['Age'].quantile(0.25)

Q3 = df['Age'].quantile(0.75)

IQR = Q3 - Q1


lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("---------------------- Handling Ouliers ----------------------")

print("Lower Boundry:",lower)
print("Upper Boundry:",upper)

outliers_found = df[(df['Age'] > upper) | (df['Age'] < lower)]

print("\nDetected Outliers\n")

print(outliers_found)
print()

df['Age'] = df['Age'].clip(lower=lower, upper=upper)

print("\nDataSet After Handling Outliers\n")

print(df)

print("\n---------------------- Handling Missing Values ----------------------")

print("Before Handling Missing Values\n")

print(df)


df = df.fillna({
    'Age': np.round(df['Age'].mean(),2),
    'Gender': df['Gender'].mode()[0],
})


print("\nAfter Handling Missing Values\n")

print(df)

print("\n---------------------- Encoding Dataset(Categorical data) ----------------------\n")

le = LabelEncoder()

# Label Encoding (Order/rank matters)

df['Grade'] = le.fit_transform(df['Grade'])

# One Hot Encoding(Order/rank doesn't Matter)

df = pd.get_dummies(df, columns=['Gender'],dtype=int)


df.drop(columns='StudentID',inplace=True)

print("\nDataset After Encoding\n")

print(df)

print("-------------------- Feature Scalling --------------------")

scaler = StandardScaler()

columns_to_scal = ['Age', 'StudyHours', 'Attendance', 'MidTerm', 'Assignment']

df[columns_to_scal] = scaler.fit_transform(df[columns_to_scal])

print("After Feature Scalling\n")
df[columns_to_scal] = df[columns_to_scal].round(2)
print(df)

print('\nMean of Sclled Features\n')

print(np.round(df[columns_to_scal].mean(),2))


print('\nSTD of Sclled Features\n')

print(np.round(df[columns_to_scal].std(),2))

X = df.drop('Passed',axis=1).values

Y = df['Passed'].values

x_train, x_test, y_train, y_test = train_test_split (
    X, Y, test_size = 0.2, random_state = 42,
    stratify=Y
)

model = LogisticRegression()

model.fit(x_train,y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)

f1Score = f1_score(y_test, predictions)

evaluation_metrices = f"Accuracy = {accuracy * 100:.2f}\nPrecision = {precision:.2f}\nRecall = {recall:.2f}\nF1 Score = {f1Score:.2f}"

print('\n----------- Evaluation Metrices -----------\n')
print(evaluation_metrices)

cm = confusion_matrix(y_test,predictions, labels = [0,1])

print('Confusion Matrix\n')
print(cm)

img = ax[2].imshow(cm, cmap = 'Blues')

plt.colorbar(img,ax=ax[2])

for i in range(2):          # loop through rows
    for j in range(2):      # loop through columns
        ax[2].text(j, i,       # x=column, y=row
                cm[i][j], # the number to show
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="black")




ax[2].set_xticks([0,1])
ax[2].set_yticks([0,1])
ax[2].set_xticklabels(['Failed', 'Passed'])
ax[2].set_yticklabels(['Failed', 'Passed'])
ax[2].set_title("Confusion Matrix",
                fontsize = 14,
                fontweight = 'bold')
ax[2].set_xlabel('Predicted',fontsize = 14, fontweight = 'bold')

ax[2].set_ylabel('Actual',fontsize = 14, fontweight = 'bold')

importance =  abs (model.coef_[0])# or [abs(model.coef_[0,0]), abs(model.coef_[0,1]), abs(model.coef_[0,2]), abs(model.coef_[0,3]), abs(model.coef_[0,4])]

features_names = df.drop('Passed',axis=1).columns.tolist()

ax[3].bar(features_names, importance, color = 'green', edgecolor = 'black')

ax[3].set_xticklabels(features_names, fontsize = 9, rotation = 45, ha = 'right')

ax[3].set_xlabel("All imp Features", color = 'black', fontsize = 12, fontweight = 'bold')

ax[3].set_ylabel("Feature Importance", color = 'black', fontsize = 12, fontweight = 'bold')

ax[3].set_title('Features VS Importance', color = 'black', fontsize = 14, fontweight = 'bold')


fig.suptitle('Student Performance Prediction System', color = 'black', fontsize = 15, fontweight = 'bold')

print("\n---------------------- Bonus Analysis ----------------------")
print("Total Passed Students:",df['Passed'].sum())
print("Total Failed Students:",len(df['Passed']) - df['Passed'].sum())
original_df = pd.DataFrame(data)
passing_students = original_df[original_df['Passed'] == 1]
failling_students = original_df[original_df['Passed'] == 0]

print("Average Study hours of passing Students:",np.round(passing_students['StudyHours'].mean(),2))
print("Average Study hours of Failing Students:",round(failling_students['StudyHours'].mean(),2))
print("By looking at the bar chart it's clear that Grade has the highest impact on passing")

plt.tight_layout()
fig.savefig('Student Grade Predictor.png', dpi = 500)

plt.show()



