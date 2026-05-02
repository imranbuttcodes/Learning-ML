from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def main():
#     # data = fetch_california_housing()
#     # df = pd.DataFrame(data.data, columns=data.feature_names)
#     # df["prices"] = data.target

#     # print(df)
#     # print(df.head())
#     # print(df.describe())
#     # print(df.info())

#     # print("Missing Values\n\n")
#     # print(df.isnull().sum())

#     # X = df.drop("prices",axis=1)

#     # Y = df["prices"]

#     # x_train,x_test,y_train,y_test = train_test_split(
#     #     X,Y,
#     #     test_size=0.2,
#     #     random_state=42,
#     #     #stratify=Y
#     # )

#     # scaler = StandardScaler()
#     # x_train = scaler.fit_transform(x_train)
#     # x_test = scaler.transform(x_test)

#     # model = LinearRegression()

#     # model.fit(x_train,y_train,)

#     # y_predictions = model.predict(x_test)

#     # print("------------ Metices Evaluation -----------------")
#     # print(f"MAE: {mean_absolute_error(y_test,y_predictions):.2%}")
#     # print(f"MSE: {mean_squared_error(y_test,y_predictions):.2%}")
#     # print(f"R^2 score: {r2_score(y_test,y_predictions):.2%}")
#     # print(pd.Series(model.coef_,index=X.columns))
#     # print(X.columns)

#   #  df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
#     df = pd.read_csv("titanic.csv")
#     target = df.pop('Survived')
#     df.insert(11,"Survived" ,target)
#     #df.to_csv("titanic.csv",index=False)

#     print(df.head())

#     print(df.isnull().sum())

#     print(df.columns)
#     print()
#     print(df.dtypes)


#     print(df["Survived"].value_counts())
#     print()
#     print(df["Survived"].value_counts(normalize=True)) # To print percentages 
#     # df = df.dropna(subset=["Embarked"])

#     # print(df.isnull().sum())


# if __name__ == "__main__":
#     main()    



df = pd.DataFrame({
    "Salary": [10000,20000,20000, 20000, 40000, 150000,2000000]
})

scl = StandardScaler() # prefer when we have outliers

df['scl_Salary'] = scl.fit_transform(df[['Salary']])

scl = MinMaxScaler() # not prefer when we have outliers

df['min_max_scl_Salary'] = scl.fit_transform(df[['Salary']])
print(df)