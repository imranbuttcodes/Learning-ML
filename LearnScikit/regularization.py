import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso, Ridge,ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing




df = fetch_california_housing()

X = pd.DataFrame(df['data'], columns=df['feature_names'])

Y = pd.DataFrame(df['target'], columns=df['target_names'])

x_train, x_test, y_train, y_test = train_test_split(
    X,Y,
    random_state=42,
    test_size=0.2
)

scl = StandardScaler()

x_train = scl.fit_transform(x_train)

x_test = scl.transform(x_test)

linear_rg = LinearRegression()




ridge = Ridge(alpha=0.1)

lasso = Lasso(alpha=0.1)

elastic_net = ElasticNet(alpha=0.01, l1_ratio=1.0)


linear_rg.fit(x_train,y_train)

y_pred_linear = linear_rg.predict(x_test)


ridge.fit(x_train, y_train)

y_pred_ridge = ridge.predict(x_test)



lasso.fit(x_train,y_train)

y_pred_lasso = lasso.predict(x_test)



elastic_net.fit(x_train,y_train)

y_pred_elastic = elastic_net.predict(x_test)

print(f"Ridge Accuracy: {r2_score(y_test,y_pred_ridge)}")
print(f"Lasso Accuracy: {r2_score(y_test,y_pred_lasso)}")
print(f"ElasticNet Accuracy: {r2_score(y_test,y_pred_elastic)}")
print(f"Linear Regression Accuracy: {r2_score(y_test,y_pred_linear)}")