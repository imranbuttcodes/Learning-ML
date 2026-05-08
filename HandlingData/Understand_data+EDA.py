"""
*Understanding Your DATA*

*First Basic Questions* We Should ask when we got dataset to train ML model on:
1. How big is the dataset(df.shape());
2. How does the dataset look like (df.head(), but preferred df.sample() due to randomness)
3. What is the data types of each cols (df.info();)
4. Are there any missing values (df.isnull().sum();)
5. How does data look like mathematically (df.describe() works only on numerical features)
6. Are there duplicate values (df.duplicated().sum();)
7. How is the correlation between cols (df.corr();)


"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic.csv")


"""
Understanding Data
"""

#Step 1: How big is data


print("STEP 1: shape of the dataset")
print(df.shape)

#Step 2: How does data look like
print("STEP 2: How does data looklike")
print(df.sample(5)) # for random sampling

#Step 3: Datatypes
print("STEP 3: datatypes")
df.info()

#step 4: Any Missing Values

print("STEP 4: Missing Values")

print(df.isnull().sum())

#Step 5: Mathematical Info about data(on Only numerical data)

print("STEP 5: Mathematicall info")

print(df.describe())

#Step 6: Duplicate values
print("STEP 6: Duplicate Vlaues")

print(df.duplicated().sum())

#Step 7: Correlation 
print("STEP 7: Correlation with target")
print(df.corr(numeric_only=True)['Survived'])


# print(df.loc[494,"Ticket"])
# print(df.loc[495,"Ticket"])

#Understanding about data is done

#Now EDA STUFF

# cat_col = df.select_dtypes(include=['object', 'category']).columns.tolist()
# num_col = df.select_dtypes(include=['number']).columns.tolist()
# print("Categorical Features")
# print(cat_col)
# print("Numerical Features")
# print(num_col)

fig, ax = plt.subplots(1, 4, figsize=(12, 5))
print(df['Survived'].value_counts())
sns.countplot(data= df, x = "Survived", color = "RED", ax = ax[0])
ax[0].set_title("Survival Distribution")
sns.countplot(data=df, x= 'Pclass',ax=ax[1])
ax[1].set_title("Pclass Distribution")
sns.countplot(data=df, x = 'Sex', ax=ax[2])
ax[2].set_title("Sex")
sns.displot(data = df, x= 'Age', ax=ax[3])
ax[3].set_title("Displot")

plt.show()



# df['Age'].plot(kind='hist')

# plt.show()