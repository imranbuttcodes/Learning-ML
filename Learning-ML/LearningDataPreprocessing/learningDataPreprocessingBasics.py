import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Messy real world data
# data = {
#     "Name":    ["Ali", "Sara", "Ahmed", "Khan", "Fatima"],
#     "Age":     [25,    np.nan, 30,      22,     np.nan],
#     "Score":   [85,    90,     np.nan,  78,     92],
#     "Grade":   ["A",   "B",    "A",     np.nan, "B"]
# }

# df = pd.DataFrame(data)
# print(df)
# print()

# # Detect missing values
# print("Before Preprocessing")
# print("Missing values in each column:")
# print(df.isnull().sum())
# print()
# print("Percentage % of missing data")
# print(df.isnull().sum()/len(df.isnull()) *100)
# print("Processing...")

# df = df.fillna({'Name':'None', 'Age': np.round(df['Age'].mean(),2), 'Grade': df['Grade'].mode()[0], 'Score': np.round(df['Score'].mean(),2)})
# print('After Preprocessing')

# print(df)

# print()
# print("Missing Values Remaining")

# print(df.isnull().sum())



"""
Quick Summary — When to Use What

Missing numerical  + no outliers  → fill with MEAN
Missing numerical  + has outliers → fill with MEDIAN
Missing categorical               → fill with MODE
Too many missing (50%+)           → drop the column
Very few missing rows             → drop the rows

"""


data = {
    "Name":  ["Ali", "Sara", "Ahmed", "Khan", "Fatima"],
    "Grade": ["A",   "B",    "A",     "C",    "B"],
    "City":  ["Lahore", "Karachi", "Lahore", "Islamabad", "Karachi"]
}

print('----------------- label Encoding -----------------')

df = pd.DataFrame(data)
print("BEFORE encoding:")
print(df)

# Label Encoding

le = LabelEncoder()

df["Grade_encoded"] = le.fit_transform(df["Grade"])
df["City_encoded"]  = le.fit_transform(df["City"])

print("\nAFTER Label Encoding:")
print(df)

print("\nGrade mapping:")
for i, cls in enumerate(le.classes_):
    print(f" {cls} -> {i}")


"""




"""
print('---------------- One Hot Encoding ----------------')
df = pd.DataFrame(data)
print('BEFORE ENCODING')
print(df)

print('After One Hot Encoding')
df = pd.get_dummies(df,columns=['City'],dtype=int)
print(df)

