import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import matplotlib as mpl

# # Set font globally
# mpl.rcParams['font.family'] = 'Times New Roman'

# # Example plot
# x = [1, 2, 3]
# y = [2, 4, 6]

# plt.plot(x, y)
# plt.title("Sample Plot")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.show()


# columns = [
#     "mpg", "cylinders", "displacement", "horsepower",
#     "weight", "acceleration", "model_year", "origin", "car_name"
# ]

# df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", na_values="?", names=columns, sep=r"\s+")

# print(df.head())
# print("Missing Values\n")
# print(df.isnull().sum())
# print("total rows:",len(df))
# print(df.shape)
# print(df.isnull().sum())

# X = df.drop(["mpg", "car_name"], axis=1)
# Y = df["mpg"]


# print(df.dtypes)

# num_col = X.select_dtypes(include=["int64", "float"]).columns.tolist()



# X_train, X_test, Y_train, Y_test = train_test_split(
#     X,Y,
#     test_size=0.2,
#     random_state=42
# )


# train_mean = X_train["horsepower"].mean()

# print(train_mean)

# X_train = X_train.fillna({
#     "horsepower": train_mean
# })

# X_test = X_test.fillna({
#     "horsepower": train_mean
# })


# scl = StandardScaler()

# X_train[num_col] = scl.fit_transform(X_train[num_col])

# X_test[num_col] = scl.transform(X_test[num_col])

# print(X_train)

# print(X_test)


# fig,ax  = plt.subplots(1,3,figsize=(8,6))

# fig.suptitle("Hello, World")

# ax[0].hist(df['mpg'], bins=5,edgecolor = "#220000")
# barlabels = ['1', '2', 3]
# print(df['origin'].value_counts())

# origin_counts = df['origin'].value_counts().sort_index()
# print(origin_counts)
# ax[1].bar(origin_counts.index.astype(str), origin_counts.values, edgecolor = "#220000")
# ax[2].hist(X_train["horsepower"], bins=5, color = "#068A27", edgecolor = "#220000")
# ax[2].set_xlabel("horsepower")
# plt.tight_layout()
# plt.show()

# model = LinearRegression()

# print("Training model...")
# model.fit(X_train, Y_train)

# predictions = model.predict(X_test)

# print("----------- Model Evaluation -------------")

# print(f"MAE: {mean_absolute_error(Y_test,predictions):.4f}")
# print(f"MSE: {mean_squared_error(Y_test,predictions):.4f}")
# print(f"RMSE: {np.sqrt(mean_squared_error(Y_test,predictions)):.4f}")
# print(f"R^2 score: {r2_score(Y_test,predictions):.4f}")


df = pd.read_csv("AmesHousing.csv")

# =============== EDA ===================== 

print(df.head())
print()
print(df.info())
print()
print(df.describe())
print()
print("Missing Values\n\n")
pd.set_option("display.max_columns", None)
print(df.isnull().sum()[df.isnull().sum() > 0])
print()
print(len(df))

print("IS PID UNIQUE: ",df["PID"].is_unique)
# so PID is unique and we also don't need it and we'll remove this as well as the features containing nAn more than 2000 as we have dataset of 2930

print(df.columns)
#pd.set_option("display.max_rows",)
print(df.dtypes.to_string())

num_col = df.select_dtypes(include=["number"]).columns.tolist()
cat_col = df.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numerical Features")
print(num_col)
print()
print("Categorical Features")
print(cat_col)

print("\nNumerical Features Missing Values: ")
print(df[num_col].isnull().sum()[df[num_col].isnull().sum() > 0])
print("\nCategorical Features Missing Values: ")
print(df[cat_col].isnull().sum()[df[cat_col].isnull().sum() > 0])

fig, ax = plt.subplots(1,6,figsize= (12,8))



ax[0].hist(df['Lot Frontage'],bins = 5, edgecolor = "black")
ax[0].set_title("Lot Frontage", fontsize = 12, fontweight= "bold")
ax[1].boxplot(df["Lot Frontage"])
ax[1].set_title("Handing Outliers")
ax[2].hist(df["Garage Yr Blt"],bins=5,edgecolor = "black")
ax[2].set_title("Garage Yr Blt", fontsize = 12, fontweight= "bold")
ax[3].boxplot(df["Garage Yr Blt"])
ax[3].set_title("Handing Outliers")

ax[4].hist(df["Mas Vnr Area"],bins=5,edgecolor = "black")
ax[4].set_title("Mas Vnr Area", fontsize = 12, fontweight= "bold")
ax[5].boxplot(df["Mas Vnr Area"])
ax[5].set_title("Mas Vnr Area")

plt.tight_layout()
plt.show()


plt.hist(df["SalePrice"], bins=5, edgecolor = "black")
plt.show()

print(df['SalePrice'].value_counts())

print("Dropping numerical features.....")

df.drop(["Order", "PID"],axis=1,inplace=True)


print("Imputing Numerical Features with median()...")

print("Droping missing numerical records")
df = df.dropna(subset=['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Garage Cars', 'Garage Area'])

print("Dropping unnecceary categorical features........")

"""
Safe to drop (mostly missing and not obviously critical for price):

Garage Type, Garage Finish, Garage Qual, Garage Cond → lots of missing
and probably heavily correlated with Garage Cars / Garage Area; you can drop and keep Garage Cars, Garage Area instead

"""
df = df.drop(["Alley", "Mas Vnr Type", "Pool QC", "Fence", "Misc Feature", "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond", "Fireplace Qu"], axis=1)


df.dropna(subset=["Electrical"], inplace=True)

print("Skewness:",df['SalePrice'].skew())

X = df.drop('SalePrice', axis=1)

Y = np.log(df["SalePrice"])

current_cat_col_after_drop = df.select_dtypes(include=['object', 'category']).columns

X[current_cat_col_after_drop] = X[current_cat_col_after_drop].fillna("None")

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,
    test_size=0.2,
  # stratify=Y
)

lf_median = X_train['Lot Frontage'].median()

GYB_median = X_train['Garage Yr Blt'].median()

X_train.fillna({
    "Lot Frontage": lf_median # because the feature is mostly positive skwed
    ,
    "Garage Yr Blt": GYB_median # because the feature is mostly positive skwed
},inplace=True)


X_test.fillna({
    "Lot Frontage": lf_median # because the feature is mostly positive skwed
    ,
    "Garage Yr Blt": GYB_median # because the feature is mostly positive skwed
},inplace=True)


# bq_mode = X_train["Bsmt Qual"].mode()[0]
# bc_mode = X_train["Bsmt Cond"].mode()[0]
# be_mode = X_train["Bsmt Exposure"].mode()[0]
# bf_t1_mode = X_train["BsmtFin Type 1"].mode()[0]
# bf_t2_mode = X_train["BsmtFin Type 2"].mode()[0]


# X_train.fillna({
#     "Bsmt Qual": bq_mode,
#     "Bsmt Cond": bc_mode,
#     "Bsmt Exposure": be_mode,
#     "BsmtFin Type 1": bf_t1_mode, 
#     "BsmtFin Type 2": bf_t2_mode 
# }
# ,inplace=True
# )


# X_test.fillna({
#     "Bsmt Qual": bq_mode,
#     "Bsmt Cond": bc_mode,
#     "Bsmt Exposure": be_mode,
#     "BsmtFin Type 1": bf_t1_mode, 
#     "BsmtFin Type 2": bf_t2_mode 
# }
# ,inplace=True
# )


# ================== Feature Scalling/encoding ========================


X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

scl = StandardScaler()

X_train = scl.fit_transform(X_train)
X_test = scl.transform(X_test)


model = LinearRegression()

model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

Y_pred_real = np.exp(Y_pred)
Y_test_real = np.exp(Y_test)

print("------------ Model Evaluation ------------")

print(f"MAE: {mean_absolute_error(Y_test_real,Y_pred_real)}")
print(f"MSE: {mean_squared_error(Y_test_real,Y_pred_real)}")
print(f"RMSE: {np.sqrt(mean_squared_error(Y_test_real,Y_pred_real))}")
print(f"r2 score: {r2_score(Y_test_real,Y_pred_real)}")

print("----------------------------------------")

