import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder


"""
## Feature Scaling

---

## The Problem

Look at this dataset:

```
Age    Salary      Score
25     50000       85
30     100000      90
22     30000       78
```

Three features — completely different scales:
```
Age    → 22 to 63        small numbers
Salary → 30000 to 100000 huge numbers
Score  → 0 to 100        medium numbers
```

---

## Why This is a Problem

ML models calculate distances and weights between features. When scales are different:

```
Salary dominates everything!
50000 vs 100000 → huge difference
Age difference of 5 → ignored completely!

Model thinks salary matters 1000x more
than age — just because of scale!
```

---

## Simple Analogy

Imagine comparing:
```
Height in meters    → 1.75
Weight in grams     → 70000

Model thinks weight is 40000x more important
just because the number is bigger!
That's obviously wrong! 😬
```

---

## Solution — Feature Scaling

Bring all features to **same scale**:

```
BEFORE scaling:
Age    → 22 to 63
Salary → 30000 to 100000
Score  → 0 to 100

AFTER scaling:
Age    → 0 to 1
Salary → 0 to 1
Score  → 0 to 1
```

Now all features are equally important! ✅

---

## Two Types of Scaling

```
MinMaxScaler    → scales between 0 and 1
StandardScaler  → scales around mean (0) and std (1)
```







"""




# data = {
#     "Age":    [25, 30, 22, 63, 35],
#     "Salary": [50000, 100000, 30000, 80000, 60000],
#     "Score":  [85, 90, 78, 92, 88]
# }

# df = pd.DataFrame(data)

# print("Before Feature Scalling...\n")

# print(df)

# #MinMaxScaler
# Scalling Formula = (value - min)/(max - min) always results from 0 to 1
# scaler = MinMaxScaler()
# scaled_df = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)

# print("After Feature Scalling (MinMaxScale)")

# print(np.round(scaled_df,2))


# data = {
#     "Name": ["Imran", "Butt", "Bro", "Code", "Why"],
#     "Age":    [25, 30, 22, 63, 35],
#     "Salary": [50000, 100000, 30000, 80000, 60000],
#     "Score":  [85, 90, 78, 92, 88]
# }

# df = pd.DataFrame(data)

# print("Before Feature Scalling...\n")

# print(df)

# #MinMaxScaler
# columns_to_scale = ['Age','Salary','Score']
# scaler = MinMaxScaler()
# #scaled_df = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
# df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# print("After Feature Scalling (MinMaxScale)")
# print(np.round(df,2))




"""
When MinMaxScaler Fails
Original data:  [25, 30, 22, 63, 35]
New data point: [150]  ← outside original range!

Scaled: (150-22)/(63-22) = 3.12 ← above 1! 😬


MinMaxScaler breaks when new data is outside training range! That's why we have StandardScaler! 🚀

Now StandardScaler — Second Type
StandardScaler uses a different formula:

scaled = (value - mean) / std

mean = average of all values
std  = standard deviation (spread of data)

Result:

Most values fall between -3 and +3
Mean becomes 0
Not limited to 0-1 range
Handles outliers better!




"""


# data = {
#     "Name": ["Imran", "Butt", "Bro", "Code", "Why"],
#     "Age":    [25, 30, 22, 63, 35],
#     "Salary": [50000, 100000, 30000, 80000, 60000],
#     "Score":  [85, 90, 78, 92, 88]
# }

# df = pd.DataFrame(data)

# print("Before StandardScaling")
# print(df)
# scaler = StandardScaler()

# df[['Age','Salary','Score']] = scaler.fit_transform(df[['Age','Salary','Score']])

# print("After StandardScalling")
# print(np.round(df,2))

# print('\nMean of each Columns (should be ~0):')

# print(np.round(df[['Age','Salary','Score']].mean(),2))

# print('Std of each column (should be ~1)')

# print(np.round(df[['Age','Salary','Score']].std(),2))


#Challange Task on till the topics covered


data = {
    "Name":       ["Ali", "Sara", "Ahmed", "Khan", "Fatima",
                   "Usman", "Zara", "Bilal", "Hina", "Omar"],

    "Age":        [20, np.nan, 22, 19, np.nan,
                   21, 23, 20, np.nan, 22],

    "City":       ["Lahore", "Karachi", "Islamabad", "Lahore", "Karachi",
                   "Lahore", np.nan, "Karachi", "Islamabad", np.nan],

    "Grade":      ["A", "B", np.nan, "C", "B",
                   "A", "B", np.nan, "C", "A"],

    "Attendance": [90, 85, 78, np.nan, 92,
                   88, np.nan, 95, 82, 89],

    "Score":      [88, 76, 82, 91, np.nan,
                   85, 79, np.nan, 83, 87],

    "Passed":     [1, 1, 1, 1, 1,
                   1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

print("First Five Rows")
print(df.head(5))
print("Shape:",df.shape)
print()
print("Missing Values in each Columns\n")

print(df.isnull().sum())

print("Missing Percentage% of each Column\n")
print(df.isnull().sum() / len(df) * 100)

#handling missing values

df = df.fillna({'Age': df['Age'].mean(), 'City': df['City'].mode()[0], 'Grade': df['Grade'].mode()[0], 'Attendance': df['Attendance'].median(),'Score':df['Score'].mean()})
print('\nData Set After Handling Missing Values\n')

print(np.round(df,2))

ave_attendance = df['Attendance'].mean()

#Encoding Data


le = LabelEncoder()

#Label Encoding (WHen rank/order Matters)

df['Grade'] = le.fit_transform(df['Grade'])

#One Hot encoding (When rank/order doesn't matter)

df = pd.get_dummies(df,columns=['City'], dtype=int)

print("\nData Set After Encoding\n")

print(df)

#Scalling
print('Before Scalling')

print(df)

s_scaler = StandardScaler()

scalled_df = df.copy()

scalled_df[['Age','Attendance','Score']] = s_scaler.fit_transform(df[['Age','Attendance','Score']])
print('\nAfter Scalling Feaures(Age, Attendance, Grade)\n')

print(np.round(scalled_df,2))

print('\nMeans of Scalled Columns(should be almost 0)\n')

print(np.round(scalled_df[['Age','Attendance','Score']].mean(),2))

print('\nSTDof Scalled Columns(should be almost 1)\n')

print(np.round(scalled_df[['Age','Attendance','Score']].std(),2))

print('\nDropping Name column as it is not useful....')
scalled_df.drop(columns=['Name'],inplace=True)

print("\nFinal Cleaned Data Set\n")
print(np.round(scalled_df,2))
print("\nShape:",scalled_df.shape)
print('\nAll columns',scalled_df.columns)

print("Most Common City:",scalled_df[['City_Islamabad' , 'City_Karachi' , 'City_Lahore']].sum().idxmax())
print('Average Attendance:',np.round(ave_attendance,2))
print('Most Appeared Grade:', np.round(scalled_df['Grade'].mode()[0]))
print('Z Score of Highest Score:', np.round(scalled_df['Score'].max(),2))


 