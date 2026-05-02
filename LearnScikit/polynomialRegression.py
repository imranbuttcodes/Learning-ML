import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



df = pd.read_csv("Weight-Height Polynomial Dataset.csv")

print(df)

x_train, x_test, y_train, y_test = train_test_split(
    df[['Weight']],
    df['Height'],
    random_state=42
)

model = LinearRegression()

model.fit(x_train,y_train)

lr_pred = model.predict(x_test)

print(f"lr r2_score: {r2_score(y_test,lr_pred)}")

x_line = np.linspace(df[['Weight']].min(), df[['Weight']].max(), 100).reshape(-1,1)

y_line = model.predict(x_line)

fig, ax = plt.subplots(1,2,figsize=(8,6))
ax[0].scatter(df['Weight'], df['Height'], color = "blue")

ax[0].plot(x_line, y_line)

ax[0].set_title("Weight/height (Using linear regression)")

ax[0].set_xlabel("Weight")
ax[0].set_ylabel("Height")


poly = PolynomialFeatures(degree=2)

X_train_trans = poly.fit_transform(x_train)

X_test_trans = poly.transform(x_test)

print(x_train.iloc[0,0])
print("HEHE")
print(X_train_trans[0])

p_model = LinearRegression()

p_model.fit(X_train_trans, y_train)

p_pred = p_model.predict(X_test_trans)


print(f"ploynomial model r2_score: {r2_score(y_test,p_pred)}")

x_line = np.linspace(df[['Weight']].min(), df[['Weight']].max(), 100).reshape(-1,1)
x_line_trans = poly.transform(x_line)
y_line = p_model.predict(x_line_trans)
ax[1].set_title("Using Polynomial Regression Model")
ax[1].scatter(df['Weight'], df['Height'], color = "blue")
ax[1].plot(x_line, y_line)
ax[1].set_xlabel("Weight")
ax[1].set_ylabel("Height")
plt.show()