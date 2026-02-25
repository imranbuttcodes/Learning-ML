import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data = {
#     "Age":   [20, 21, 22, 19, 23, 20, 22, 21, 999, 20, 200],
#     "Score": [88, 76, 82, 91, 85, 79, 83, 87, 80,  92, 200]
# }

# df = pd.DataFrame(data)

# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# ax[0].boxplot(df["Age"])
# ax[0].set_title("Age — Outlier Visible!", fontweight="bold")
# ax[0].set_ylabel("Age")

# ax[1].boxplot(df["Score"])
# ax[1].set_title("Score —  Outliers Visible", fontweight="bold")
# ax[1].set_ylabel("Score")

# plt.tight_layout()
# plt.show()




data = {
    "Age":   [20, 21, 22, 19, 23, 20, 22, 21, 999, 20],
    "Score": [88, 76, 82, 91, 85, 79, 83, 87, 80,  92]
}

df = pd.DataFrame(data)

# Calculate IQR for Age
Q1  = df["Age"].quantile(0.25)
Q3  = df["Age"].quantile(0.75)
IQR = Q3 - Q1

print(f"Q1  = {Q1}")
print(f"Q3  = {Q3}")
print(f"IQR = {IQR}")

# Calculate boundaries
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"\nLower boundary = {lower}")
print(f"Upper boundary = {upper}")

# Find outliers
outliers = df[df["Age"] > upper]
print(f"\nOutliers found:")
print(outliers)


# Handling Outliers 

# Method 1

# df_clean = df[df["Age"] <= upper]
# print("After removing outlier:")
# print(df_clean)

#Method 2

#Replace with median

# median_age = df["Age"].median()
# print(median_age)

# print("Before\n",df)

# # df["Age"] = df["Age"].apply(
# #     lambda x: median_age if x > upper else x
# # )


# #OR through
# df.loc[df['Age'] > upper, 'Age'] = median_age
# print("After replacing outlier with median:")
# print(df)

# Option 3 — Replace with boundary value
print(upper,lower)
df["Age"] = df["Age"].clip(lower=lower, upper=upper)
print("After clipping outlier:")
print(df)

"""
## When to Use Which
```
Remove row    → when entire row is bad data
               → dataset is large enough

Replace with  → when you want to keep the row
median        → dataset is small

Clip          → cap at boundary value
               → quick and simple fix

"""
