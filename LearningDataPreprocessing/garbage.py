import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score 





# diabetes_df = pd.read_csv('diabetes.csv')

# fig, ax = plt.subplots(1,3,figsize = (18,6))
# class_counts = diabetes_df['Outcome'].value_counts()
# print(class_counts)
# ax[0].bar(['Non Diabetic', 'Diabetic'],class_counts, color = ['#2563EB', '#DC2626'], edgecolor = 'black')
# ax[0].set_title('Class Distribution',fontweight='bold', fontsize=12)

# ax[0].set_xlabel('Class', fontweight='bold')
# ax[0].set_ylabel('Count', fontweight='bold')
# for i, val in enumerate(class_counts.values):
#     ax[0].text(i, val + 5, str(val),
#         ha='center', fontweight='bold')


# ax[1].hist(diabetes_df['Glucose'], bins=20,
#     color='#2563EB', edgecolor='black')
# ax[1].set_title('Glucose Distribution',
#             fontweight='bold', fontsize=12)
# ax[1].set_xlabel('Glucose', fontweight='bold')
# ax[1].set_ylabel('Count', fontweight='bold')
# ax[1].axvline(diabetes_df['Glucose'].mean(),
#         color='red', linestyle='--',
#         linewidth=2, label='Mean')
# ax[1].legend()

# diabetic     = diabetes_df[diabetes_df['Outcome'] == 1]
# non_diabetic = diabetes_df[diabetes_df['Outcome'] == 0]

# ax[2].scatter(non_diabetic['Glucose'],
#         non_diabetic['BMI'],
#         color='#2563EB', edgecolor='black',
#         s=30, alpha=0.6, label='Not Diabetic')
# ax[2].scatter(diabetic['Glucose'],
#         diabetic['BMI'],
#         color='#DC2626', edgecolor='black',
#         s=30, alpha=0.6, label='Diabetic')
# ax[2].set_title('BMI vs Glucose',
#             fontweight='bold', fontsize=12)
# ax[2].set_xlabel('Glucose', fontweight='bold')
# ax[2].set_ylabel('BMI', fontweight='bold')
# ax[2].legend()

# plt.tight_layout()
# plt.savefig('01_data_overview.png', dpi=300)
# plt.show()



diabetes_df = pd.read_csv('diabetes.csv')

fig, ax = plt.subplots(1,3,figsize = (18,6))
class_counts = diabetes_df['Outcome'].value_counts()
print(class_counts)
ax[0].bar(['Non Diabetic', 'Diabetic'],class_counts, color = ['#2563EB', '#DC2626'], edgecolor = 'black')
ax[0].set_title('Class Distribution',fontweight='bold', fontsize=12)

ax[0].set_xlabel('Class', fontweight='bold')
ax[0].set_ylabel('Count', fontweight='bold')
for i, val in enumerate(class_counts.values):
    ax[0].text(i, val + 5, str(val),
                ha='center', fontweight='bold')


ax[1].hist(diabetes_df['Glucose'], bins=20,
            color='#2563EB', edgecolor='black')
ax[1].set_title('Glucose Distribution',
                fontweight='bold', fontsize=12)
ax[1].set_xlabel('Glucose', fontweight='bold')
ax[1].set_ylabel('Count', fontweight='bold')
ax[1].axvline(diabetes_df['Glucose'].mean(),
            color='red', linestyle='--',
            linewidth=2, label='Mean')
ax[1].legend()

diabetic     = diabetes_df[diabetes_df['Outcome'] == 1]
non_diabetic = diabetes_df[diabetes_df['Outcome'] == 0]

ax[2].scatter(non_diabetic['Glucose'],
            non_diabetic['BMI'],
            color='#2563EB', edgecolor='black',
            s=30, alpha=0.6, label='Not Diabetic')
ax[2].scatter(diabetic['Glucose'],
            diabetic['BMI'],
            color='#DC2626', edgecolor='black',
            s=30, alpha=0.6, label='Diabetic')
ax[2].set_title('BMI vs Glucose',
                fontweight='bold', fontsize=12)
ax[2].set_xlabel('Glucose', fontweight='bold')
ax[2].set_ylabel('BMI', fontweight='bold')
ax[2].legend()

plt.tight_layout()
#plt.savefig('01_data_overview.png', dpi=300)
plt.show()