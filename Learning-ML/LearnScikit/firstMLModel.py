import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# scores = np.array([35, 45, 50, 58, 65, 70, 75, 82, 88, 95])

# # How many samples do we have?
# print("Total samples:", len(hours))

# # What are min and max values?
# print("Min hours:", hours.min(), "Max hours:", hours.max())
# print("Min score:", scores.min(), "Max score:", scores.max())

# # Plot to see the relationship visually
# fig, ax = plt.subplots(figsize=(8, 5))

# ax.scatter(hours, scores, color="steelblue",
#            edgecolor="black", s=100)
# m,b = np.polyfit(hours,scores,1)
# y = [m*x + b for x in hours]
# ax.plot(hours,y,linestyle = '-',linewidth = 2, color ='green',label = 'Best Fit line')
# ax.set_title("Hours Studied vs Exam Score",
#              fontsize=14, fontweight="bold")
# ax.set_xlabel("Hours Studied")
# ax.set_ylabel("Exam Score")
# ax.grid(axis="both", linestyle="--", alpha=0.5)
# ax.legend()

# plt.tight_layout()
# plt.show()


hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
scores = np.array([35, 45, 50, 58, 65, 70, 75, 82, 88, 95])

X = hours.reshape(-1,1)
Y = scores

x_train , x_test , y_train, y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state = 45
)
model = LinearRegression()
model.fit(x_train,y_train)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
print("Model trained successfully!")
print(f"Slop (m):{model.coef_[0]:.2f}")
print(f"Intercept:{model.intercept_:.2f}")
print(f"Equation: score = {model.coef_[0]:.2f} x hours + {model.intercept_:.2f}")

# Manual prediction for 5 hours
manual = 6.42 * 5 + 30.80
print(f"Manual prediction for 5 hours: {manual:.2f}")

prediction = model.predict(x_test)
print(f'Test hours: {x_test.flatten()}')
print(f'Actual Scores: {y_test}')
print(f'Predictions: {prediction}')