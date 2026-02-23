import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Features
glucose     = [85,  166, 77,  183, 116, 103, 125, 110, 145, 158,
               72,  140, 95,  154, 168, 88,  130, 102, 175, 120]

bmi         = [26.6, 19.9, 27.1, 23.3, 28.7, 43.3, 31.6, 35.3, 29.9, 32.1,
               25.4, 30.2, 22.8, 28.9, 33.1, 27.5, 29.8, 24.6, 31.2, 28.3]

age         = [31, 21, 29, 63, 22, 33, 47, 28, 40, 35,
               24, 45, 26, 52, 38, 30, 42, 27, 55, 36]

# Label (1 = Diabetic, 0 = Not Diabetic)
diabetic    = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1,
               0, 1, 0, 1, 1, 0, 1, 0, 1, 0]

df = pd.DataFrame({'Glucose': glucose,
                   'BMI': bmi,
                   'age': age,
                   'diabetic': diabetic})
print(df.head(5))
counts = df['diabetic'].value_counts()
print('Total diabetic: ',counts[1])
print('Total Non diabetic',counts[0])

fig, ax = plt.subplots(1,3,figsize = (18,6))

ax[0].scatter(glucose,diabetic,color = 'blue',edgecolor = 'black',s=100)

ax[0].set_title("GLucose VS Diabetic", color = 'black',fontsize = 12,fontweight = 'bold')
ax[0].set_xlabel("Glucos",color = 'black', fontsize =12, fontweight = 'bold')
ax[0].set_ylabel("Diabetics",color = 'black', fontsize =12, fontweight = 'bold')

X = df.drop('diabetic',axis=1).values

Y = np.array(df['diabetic'].tolist())

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=42,
)

model = LogisticRegression()

model.fit(x_train,y_train)

predictions = model.predict(x_test)
probs = model.predict_proba(x_test)
df = pd.DataFrame(x_test,columns=['Glucose', 'BMI', 'age'])
df['Actual Diabetics'] = pd.Series(y_test)
df['Predicted Diabetics'] = pd.Series(predictions)
df['Actual Diabetics']= df['Actual Diabetics'].astype('bool')
df['Predicted Diabetics'] = df['Predicted Diabetics'].astype('bool')
# df['Predicted Probabilities'] = pd.Series(probs)
print('Test features')
print(df)
print()
print("Predicted Probabilites of getting non-diabities vs diabetics")
probaiblity_df = pd.DataFrame(np.round(probs,2),columns=['Not Diabetics %', 'Diabetics%'])
print(probaiblity_df)

print(f"Acurrance: {accuracy_score(y_test,predictions):.2f}")
print(f"Precision: {precision_score(y_test,predictions):.2f}")
print(f"Recall Score: {recall_score(y_test,predictions):.2f}")
print(f"F1 Score: {f1_score(y_test,predictions):.2f}")
print("Confusion Matrix")
cm = confusion_matrix(y_test,predictions)

print(cm)

im = ax[1].imshow(cm,cmap='Blues')

plt.colorbar(im)

ax[1].set_title(f"Confusion Matrix\nAccuracy {accuracy_score(y_test,predictions)*100:.2f}%",fontsize = 12,fontweight = 'bold')

ax[1].set_xticks([0,1])
ax[1].set_yticks([0,1])
ax[1].set_xticklabels(['Not diabetic','diabetic'])
ax[1].set_yticklabels(['Not diabetic','diabetic'])
ax[1].set_xlabel('Predicted',fontsize = 12,fontweight = 'bold')
ax[1].set_ylabel('Actual',fontsize = 12,fontweight = 'bold')

for i in range(2):
    for j in range(2):
        ax[1].text(
            j,i,
            cm[i][j],
            color ='white' if cm[i][j] > 1 else 'black',
            fontsize = 14,
            fontweight = 'bold',
            ha = 'center',
            va = 'center'
        )

print("Testing on New custom data")

new_patients = [
    [120, 28.5, 35],   # glucose, bmi, age
    [180, 35.0, 55],
    [90,  24.0, 25],
]

new_predictions = model.predict(new_patients)

new_probs = model.predict_proba(new_patients)

new_df = pd.DataFrame(new_patients,columns=['Glucose', 'BMI', 'Age'])


new_df['Diabetics(predicted)'] = pd.Series(new_predictions)

print(new_df)
print()

print("Predicted Probabilites of getting non-diabities vs diabetics('Its on new data)")
probaiblity_df = pd.DataFrame(np.round(new_probs,2),columns=['Not Diabetics %', 'Diabetics%'])
print(probaiblity_df)
#print(df)
df = pd.DataFrame({'Glucose': glucose,
                   'BMI': bmi,
                   'age': age,
                   'diabetic': diabetic})

diabetic_patients = df[df['diabetic'] == True]

non_diabetic_patients = df[df['diabetic'] ==False]

ax[2].scatter(diabetic_patients['Glucose'],diabetic_patients['BMI'],color = 'r', edgecolor = "#C83A3A", s= 100, label = 'Diabetics')

ax[2].scatter(non_diabetic_patients['Glucose'], non_diabetic_patients['BMI'], color = 'blue', edgecolor = "#796FE7", s = 100, label = 'Non Diabetic')

ax[2].set_title("Diabetics VS Non-Diabetics Patients", fontweight = 'bold', fontsize = 14)

ax[2].set_xlabel('Glucose', fontweight = 'bold', fontsize = 12)

ax[2].set_ylabel('BMI', fontweight = 'bold', fontsize = 12)
ax[2].legend(fontsize = 12)
fig.suptitle('Diabetes Prediction System — Logistic Regression', fontsize = 14, fontweight = 'bold')
print("\nBorderline Patients (40%-60% probability):")
# print(probs)
for i, prob in enumerate(new_probs):
    if 0.40 <= prob[1] <= 0.60:
        print(f"Patient {i+1} -> {prob[1]*100:.1f}% diabetic probability → BORDERLINE!")
    else:
        print(f"Patient {i+1} -> {prob[1]*100:.1f}% → clear prediction")

fig.savefig('Diabetes Prediction System_Logistic Regression.jpg',dpi = 150)

plt.tight_layout()
plt.show()


