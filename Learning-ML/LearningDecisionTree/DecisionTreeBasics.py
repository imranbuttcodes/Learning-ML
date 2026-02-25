import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree


iris = load_iris()

X = iris.data

Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split (
    X, 
    Y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion='entropy',   # or 'entropy'
    max_depth=3,        # limit depth (optional)
    random_state=42,
    min_samples_split=10,
    min_samples_leaf=5,
)

model.fit(X_train, Y_train)
predictions = model.predict(X_test)

cm = confusion_matrix(Y_test, predictions)
print(cm)
print(f"Testing Accuracy Accuracy: {accuracy_score(Y_test, predictions) * 100}%")

train_prediction = model.predict(X_train)

print(f"Traing Accuracy: {accuracy_score(Y_train, train_prediction) * 100}%")
# fig, ax = plt.subplots(figsize = (12,8))

# tree.plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled= True)
# print(classification_report(Y_test, predictions, target_names=iris.target_names))

df = pd.DataFrame(X, columns=iris.feature_names)

df['Species'] = Y

print(df)

# plt.show()


