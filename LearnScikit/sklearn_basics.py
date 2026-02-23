import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
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


# hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# scores = np.array([35, 45, 50, 58, 65, 70, 75, 82, 88, 95])

# X = hours.reshape(-1,1)
# Y = scores

# x_train , x_test , y_train, y_test = train_test_split(
#     X,
#     Y,
#     test_size = 0.2,
#     random_state = 45 # This number is just the convention you can use any number
# )
# model = LinearRegression()
# model.fit(x_train,y_train)
# # print(x_train)
# # print(x_test)
# # print(y_train)
# # print(y_test)

# print(x_test.flatten())
# print(y_test.flatten())
# print("Model trained successfully!")
# print(f"Slop (m):{model.coef_[0]:.2f}")
# print(f"Intercept:{model.intercept_:.2f}")
# print(f"Equation: score = {model.coef_[0]:.2f} x hours + {model.intercept_:.2f}")

# # Manual prediction for 5 hours
# manual = 6.42 * 5 + 30.80
# print(f"Manual prediction for 5 hours: {manual:.2f}")

# prediction = model.predict(x_test)
# print(f'Test hours: {x_test.flatten()}')
# print(f'Actual Scores: {y_test}')
# print(f'Predictions: {prediction}')


# study_hours     = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
#                    3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

# exam_scores     = [45, 55, 62, 70, 78, 82, 87, 91, 94, 97,
#                    48, 58, 66, 73, 80, 84, 89, 92, 95, 98]

# X = np.array(study_hours)
# X = X.reshape(-1,1)
# Y = np.array(exam_scores)

# print(f'Total Samples: {len(X)}')

# print(f"Min study_hours: {min(X.flatten())}")
# print(f"Max study_hours: {max(X.flatten())}")

# print(f"Min exam_scores: {min(Y)}")
# print(f"Max exam_scores: {max(Y)}")

# fig,ax = plt.subplots(figsize=(12,8))
# ax.grid(axis='both',linestyle ='--', color='gray')

# ax.scatter(X.flatten(),Y,edgecolors='black',color = 'green',s=100)


# ax.set_title('Hours Studied vs Exam Score',fontweight='bold', color='Black')
# ax.set_xlabel('Hours Studied', fontweight='bold',color='black',fontsize = 12)
# ax.set_ylabel('Exam Score',fontweight='bold',color='black', fontsize = 12)

# x_train, x_test, y_train, y_test  = train_test_split(
#     X,
#     Y,
#     random_state=45,
#     test_size=0.2,
# )

# print(f'Total Samples of training data: {len(x_train)}')
# print(f"Total Samples in test data: {len(x_test)}")

# model = LinearRegression()

# model.fit(x_train,y_train)

# print("Model trained Successfully!")
# print(f"Slop: {model.coef_[0]}")
# print(f'Intercept: {model.intercept_}')

# print(f'Equation: {model.coef_[0]} * hours_studied + {model.intercept_}')

# print("Testing out Model")
# prediction = model.predict(x_test)
# print(f'Hours stidied: {x_test.flatten()}')
# print(f'Actual Scores: {y_test}')
# print(f'Predicted Scores: {np.round(prediction,2)}')

# print('-------- End --------')

# y = [model.coef_[0]*x + model.intercept_ for x in X.flatten()]
# ax.plot(X.flatten(), y, linewidth = 2,label='Best Fit')
# ax.legend(facecolor = "#f0b6b6ff")
# plt.tight_layout()
# plt.show()


#Learning MAE(Mean Absolute Error)

# actual    = [85, 70, 90]
# predicted = [83, 75, 86]

# errors = []
# for a, p in zip(actual, predicted):
#     error = abs(a - p)      # abs() removes negative sign
#     errors.append(error)
#     print(f"Actual: {a}, Predicted: {p}, Error: {error}")

# mae = sum(errors) / len(errors)
# print(f"\nMAE = {mae:.2f}")

# hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# scores = np.array([35, 45, 50, 58, 65, 70, 75, 82, 88, 95])

# X = hours.reshape(-1,1)
# Y = scores

# model = LinearRegression()

# x_train, x_test, y_train, y_test = train_test_split(
#     X,
#     Y,
#     test_size=0.2,
#     random_state= 42,
# )

# model.fit(x_train,y_train)

# predictions = model.predict(x_test)

# mae = mean_absolute_error(y_test,predictions) 
# mse = mean_squared_error(y_test, predictions)
# rmse = np.sqrt(mse)
# print(f'Actual: {y_test}')
# print(f'Predicted: {np.round(predictions,2)}')
# print(f"MAE: {round(mae,2)}")
# print(f"MSE: {round(mse,2)}")
# print(f'RMSE: {round(rmse,2)}')

"""
## MAE vs RMSE — Which to Use?
```
MAE  → treats all errors equally
RMSE → punishes big errors more
```

Example:
```
Predictions: [off by 1, off by 1, off by 10]

MAE  = (1+1+10)/3  = 4.0  ← looks okay
RMSE = √(1+1+100)/3 = 5.8 ← catches the big error!
```

### When to use which:
```
MAE  → when all errors are equally important
RMSE → when big errors are very bad
       like medical diagnosis, stock prices

"""

#R^2 Score

# actual    = [35, 45, 50, 58, 65, 70, 75, 82, 88, 95]
# predicted = [37, 43, 52, 56, 63, 72, 77, 80, 90, 93]

# # Average of actual scores
# average = sum(actual) / len(actual)
# print(f"Average score: {average}")

# # How wrong is dumb model?
# dumb_errors = [(a - average)**2 for a in actual]
# dumb_mse    = sum(dumb_errors) / len(dumb_errors)
# print(f"Dumb model MSE: {dumb_mse:.2f}")

# # How wrong is our model?
# our_errors = [(a - p)**2 for a, p in zip(actual, predicted)]
# our_mse    = sum(our_errors) / len(our_errors)
# print(f"Our model MSE: {our_mse:.2f}")

# # R² = how much better are we than dumb model?
# r2 = 1 - (our_mse / dumb_mse)
# print(f"\nR² Score: {r2:.2f}")
# print(f"Our model explains {r2*100:.1f}% of the data!")


hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
scores = np.array([35, 45, 50, 58, 65, 70, 75, 82, 88, 95])

X = hours.reshape(-1, 1)
y = scores

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# All 4 metrics together
mae  = mean_absolute_error(y_test, predictions)
mse  = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, predictions)

print("=" * 40)
print("       MODEL EVALUATION REPORT")
print("=" * 40)
print(f"Actual scores:    {y_test}")
print(f"Predicted scores: {np.round(predictions, 2)}")
print("-" * 40)
print(f"MAE:  {mae:.2f}  → off by {mae:.2f} points on average")
print(f"MSE:  {mse:.2f}  → squared error")
print(f"RMSE: {rmse:.2f}  → off by {rmse:.2f} points on average")
print(f"R²:   {r2:.2f}   → model explains {r2*100:.1f}% of data")
print("=" * 40)


# ---

# ## What Good Results Look Like
# ```
# MAE  → as LOW as possible
# MSE  → as LOW as possible
# RMSE → as LOW as possible
# R²   → as HIGH as possible (close to 1.0)




# # 1. Setup Data
# hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# scores = np.array([35, 45, 50, 58, 65, 70, 75, 82, 88, 95])

# X = hours.reshape(-1, 1)
# y = scores

# # 2. Split and Train
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = LinearRegression()
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)

# # 3. Calculate Metrics (Converted to Python floats)
# mae = float(mean_absolute_error(y_test, predictions))
# mse = float(mean_squared_error(y_test, predictions))
# rmse = float(np.sqrt(mse))
# r2 = float(r2_score(y_test, predictions))
# slope = float(model.coef_[0])
# intercept = float(model.intercept_)

# print(f"✅ Model trained! R²: {r2:.3f}")
# print(f"Equation: y = {slope:.2f}x + {intercept:.2f}")

# # 4. Visualization
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.set_facecolor("#F8FAFC")
# fig.set_facecolor("#F8FAFC")

# # Scatter plots
# ax.scatter(X_train, y_train, color="steelblue", edgecolor="black",
#            s=100, label="Training Data", zorder=3)
# ax.scatter(X_test, y_test, color="green", edgecolor="black",
#            s=100, label="Test Data", zorder=3)

# # Regression Line
# line_x = np.linspace(0, 11, 100)
# print("Line Space X:",line_x)
# line_y = slope * line_x + intercept
# ax.plot(line_x, line_y, color="red", linewidth=2,
#         linestyle="--", label="Regression Line")

# # FIX: Ensure scalars for annotations to avoid TypeError
# for i in range(len(X_test)):
#     x_val = X_test[i].item()      # Convert from [val] to val
#     y_actual = y_test[i].item()   # Convert from [val] to val
#     y_pred = predictions[i].item() # Convert from [val] to val
    
#     ax.annotate("", 
#                 xy=(x_val, y_actual), 
#                 xytext=(x_val, y_pred),
#                 arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))

# # Metrics Box
# metrics_text = f"MAE: {mae:.1f}\nMSE: {mse:.1f}\nRMSE: {rmse:.1f}\nR²: {r2:.3f}"
# ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
#         fontsize=11, verticalalignment="top",
#         bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

# # Labels and Layout
# ax.set_title("Linear Regression — Hours Studied vs Exam Score", fontsize=14, fontweight="bold")
# ax.set_xlabel("Hours Studied", fontsize=12)
# ax.set_ylabel("Exam Score", fontsize=12)
# ax.legend(fontsize=11)
# ax.grid(True, linestyle="--", alpha=0.5)

# plt.tight_layout()
# plt.subplots_adjust(top=0.92) 
# plt.show()

# print("🎉 PLOT WORKS PERFECTLY!")




# fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# # Without linspace — jagged
# x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y1 = [x**2 for x in x1]
# ax[0].plot(x1, y1, color="red", linewidth=2)
# ax[0].set_title("Without linspace — Jagged")

# # With linspace — smooth
# x2 = np.linspace(1, 10, 100)
# y2 = x2**2
# ax[1].plot(x2, y2, color="red", linewidth=2)
# ax[1].set_title("With linspace — Smooth")

# plt.tight_layout()
# plt.show()



# fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# # Left — data coordinates
# ax[0].plot([1,2,3], [100000, 200000, 300000])
# ax[0].text(1, 80000, "I'm lost!",
#            fontsize=12, color="red")
# ax[0].set_title("Data Coordinates — Unpredictable!")

# # Right — transAxes coordinates
# ax[1].plot([1,2,3], [100000, 200000, 300000])
# ax[1].text(0.02, 0.97, "Always top left!",
#            transform=ax[1].transAxes,
#            fontsize=12, color="red",
#            verticalalignment="top")
# ax[1].set_title("transAxes — Always in place!")

# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(figsize=(8, 6))
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)

# # Different boxstyles
# styles = ["round", "square", "round4", "sawtooth", "roundtooth"]
# colors = ["#E2E8F0", "#FEF3C7", "#DCFCE7", "#FCE7F3", "#EDE9FE"]

# for i, (style, color) in enumerate(zip(styles, colors)):
#     ax.text(5, 9 - i * 1.8,
#             f'boxstyle="{style}"',
#             ha="center",
#             fontsize=12,
#             bbox=dict(boxstyle=style,
#                      facecolor=color,
#                      edgecolor="gray",
#                      alpha=0.9))

# ax.set_title("bbox boxstyle options", fontweight="bold")
# ax.axis("off")

# plt.tight_layout()
# plt.show()

"""
## `bbox=dict()` — Adding a Box Around Text

`bbox` stands for **bounding box**. It adds a colored box behind your text to make it more readable.

---

## Without bbox — text floats on chart

```python
ax.text(0.02, 0.97, "MAE: 1.31",
        transform=ax.transAxes)
```

Text just floats on the chart — hard to read if there's data behind it.

---

## With bbox — text has a box behind it

```python
ax.text(0.02, 0.97, "MAE: 1.31",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round",
                  facecolor="white",
                  alpha=0.8))
```

Now text sits in a clean white box — always readable! ✅

---

## bbox options

```python
bbox=dict(
    boxstyle="round",      # shape of box
    facecolor="white",     # background color
    edgecolor="black",     # border color
    alpha=0.8              # transparency
)
```

### `boxstyle` options:
```python
boxstyle="round"        # rounded corners
boxstyle="square"       # sharp corners
boxstyle="round4"       # more rounded
boxstyle="sawtooth"     # zigzag edges
boxstyle="roundtooth"   # rounded zigzag
```

---

## See them all

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Different boxstyles
styles = ["round", "square", "round4", "sawtooth", "roundtooth"]
colors = ["#E2E8F0", "#FEF3C7", "#DCFCE7", "#FCE7F3", "#EDE9FE"]

for i, (style, color) in enumerate(zip(styles, colors)):
    ax.text(5, 9 - i * 1.8,
            f'boxstyle="{style}"',
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle=style,
                     facecolor=color,
                     edgecolor="gray",
                     alpha=0.9))

ax.set_title("bbox boxstyle options", fontweight="bold")
ax.axis("off")

plt.tight_layout()
plt.show()
```

---

"""




# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# # Left — scatter behind line
# ax[0].plot(x, y, color="red", linewidth=4, zorder=2)
# ax[0].scatter(x, y, color="blue", s=100, zorder=1)  # behind line
# ax[0].set_title("Scatter Behind Line (zorder=1)")

# # Right — scatter in front of line
# ax[1].plot(x, y, color="red", linewidth=4, zorder=1)
# ax[1].scatter(x, y, color="blue", s=100, zorder=2)  # in front of line
# ax[1].set_title("Scatter In Front (zorder=2)")

# plt.tight_layout()
# plt.show()



# hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# scores = np.array([35, 45, 50, 58, 65, 70, 75, 82, 88, 95])

# X = hours.reshape(-1, 1)
# y = scores

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# model = LinearRegression()
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)

# mae  = mean_absolute_error(y_test, predictions)
# mse  = mean_squared_error(y_test, predictions)
# rmse = np.sqrt(mse)
# r2   = r2_score(y_test, predictions)

# # ── VISUALIZATION ──────────────────────────
# fig, ax = plt.subplots(figsize=(10, 6))
# fig.set_facecolor("#F8FAFC")
# ax.set_facecolor("#F8FAFC")

# # Training data — blue dots
# ax.scatter(X_train, y_train,
#            color="steelblue",
#            edgecolor="black",
#            s=100,
#            label="Training Data",
#            zorder=3)          # on top of everything

# # Test data — green dots
# ax.scatter(X_test, y_test,
#            color="green",
#            edgecolor="black",
#            s=100,
#            label="Test Data",
#            zorder=3)          # on top of everything

# # Regression line — smooth using linspace
# line_x = np.linspace(1, 10, 100)   # 100 smooth points
# line_y = model.coef_[0] * line_x + model.intercept_
# ax.plot(line_x, line_y,
#         color="red",
#         linewidth=2,
#         linestyle="--",
#         label="Regression Line",
#         zorder=2)             # behind dots

# # Purple arrows — show error between actual and predicted
# for i in range(len(X_test)):
#     ax.annotate("",
#                 xy=(X_test[i][0],     y_test[i]),
#                 xytext=(X_test[i][0], predictions[i]),
#                 arrowprops=dict(arrowstyle="<->",
#                                color="purple",
#                                lw=1.5),
#                 zorder=4)     # on top of everything

# # Metrics box — always top left using transAxes
# metrics_text = f"MAE:  {mae:.2f}\nMSE:  {mse:.2f}\nRMSE: {rmse:.2f}\nR²:   {r2:.2f}"
# ax.text(0.02, 0.97,
#         metrics_text,
#         transform=ax.transAxes,      # percentage coordinates
#         fontsize=11,
#         verticalalignment="top",
#         fontfamily="monospace",
#         bbox=dict(boxstyle="round",  # rounded box
#                   facecolor="#E2E8F0",
#                   alpha=0.8),
#         zorder=5)                    # topmost layer

# ax.set_title("Linear Regression — Hours Studied vs Exam Score",
#              fontsize=14, fontweight="bold")
# ax.set_xlabel("Hours Studied", fontsize=12)
# ax.set_ylabel("Exam Score", fontsize=12)
# ax.legend(fontsize=11)
# ax.grid(axis="both", linestyle="--", alpha=0.5)

# plt.tight_layout()
# plt.show()

# ## What You'll See
# ```
# Blue dots   → training data points
# Green dots  → test data points
# Red line    → regression line (smooth!)
# Purple arrows → error between actual and predicted
# Top left box  → all 4 metrics neatly displayed

