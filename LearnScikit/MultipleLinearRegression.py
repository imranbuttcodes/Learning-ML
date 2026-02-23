import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# hours      = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10]
# attendance = [60, 65, 70, 75, 80, 85, 90, 92, 95, 98]
# prev_score = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
# scores     = [35, 45, 50, 58, 65, 70, 75, 82, 88, 95]

# # Put all features together in a DataFrame
# df = pd.DataFrame({
#     "Hours"      : hours,
#     "Attendance" : attendance,
#     "Prev Score" : prev_score,
#     "Score"      : scores
# })

# print(df)



# hours      = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10]
# attendance = [60, 65, 70, 75, 80, 85, 90, 92, 95, 98]
# prev_score = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
# scores     = [35, 45, 50, 58, 65, 70, 75, 82, 88, 95]

# # Put all features into one 2D array
# X = np.array(list(zip(hours, attendance, prev_score)))
# y = np.array(scores)

# print("X shape:", X.shape)   # should be (10, 3) — 10 rows, 3 features
# print("y shape:", y.shape)   # should be (10,)  — 10 labels
# print()
# print("First 3 rows of X:")
# print(X[:3])

# # Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Train — exact same as before!
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict
# predictions = model.predict(X_test)

# # Evaluate
# mae  = mean_absolute_error(y_test, predictions)
# mse  = mean_squared_error(y_test, predictions)
# rmse = np.sqrt(mse)
# r2   = r2_score(y_test, predictions)

# print("=" * 45)
# print("       MULTIPLE LINEAR REGRESSION")
# print("=" * 45)
# print(f"Coefficients:")
# print(f"  Hours      → {model.coef_[0]:.4f}")
# print(f"  Attendance → {model.coef_[1]:.4f}")
# print(f"  Prev Score → {model.coef_[2]:.4f}")
# print(f"  Intercept  → {model.intercept_:.4f}")
# print("-" * 45)
# print(f"Equation:")
# print(f"score = {model.coef_[0]:.2f}×hours + {model.coef_[1]:.2f}×attendance + {model.coef_[2]:.2f}×prev_score + {model.intercept_:.2f}")
# print("-" * 45)
# print(f"Actual scores:    {y_test}")
# print(f"Predicted scores: {np.round(predictions, 2)}")
# print("-" * 45)
# print(f"MAE:  {mae:.2f}")
# print(f"MSE:  {mse:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"R²:   {r2:.2f}")
# print("=" * 45)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── DATA ────────────────────────────────────────────────────────────────────
size        = [650,  780,  1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000,
               700,  850,  1100, 1300, 1600, 1900, 2100, 2300, 2600, 3200]

bedrooms    = [1, 2, 2, 3, 3, 3, 4, 4, 4, 5,
               1, 2, 2, 3, 3, 3, 4, 4, 4, 5]

age         = [10, 8, 15, 5, 12, 3, 7, 20, 2, 1,
               12, 6, 18, 4, 10, 2, 8, 15, 3, 1]

prices      = [150000, 180000, 200000, 250000, 300000,
               350000, 400000, 420000, 480000, 550000,
               160000, 190000, 210000, 260000, 310000,
               360000, 410000, 430000, 490000, 560000]

# ── TASK 1 — UNDERSTAND DATA ─────────────────────────────────────────────────
data = np.array(list(zip(size, bedrooms, age, prices)))
df   = pd.DataFrame(data, columns=["Size", "Bedrooms", "Age", "Prices"])

print("=" * 50)
print("           DATASET OVERVIEW")
print("=" * 50)
print("\nFirst 5 rows:")
print(df.head())
print(f"\nMin Price:  ${df['Prices'].min():,.0f}")
print(f"Max Price:  ${df['Prices'].max():,.0f}")
print(f"Mean Price: ${df['Prices'].mean():,.0f}")

# ── TASK 2 — PREPARE AND SPLIT ───────────────────────────────────────────────
X = df.drop("Prices", axis=1).values
y = np.array(df["Prices"].tolist())

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── TASK 3 — TRAIN MODEL ─────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

size_m    = float(model.coef_[0])
bedroom_m = float(model.coef_[1])
age_m     = float(model.coef_[2])
intercept = float(model.intercept_)

print("\n" + "=" * 50)
print("           COEFFICIENTS")
print("=" * 50)
print(f"Size      → {size_m:.4f}")
print(f"Bedrooms  → {bedroom_m:.4f}")
print(f"Age       → {age_m:.4f}")
print(f"Intercept → {intercept:.4f}")
print(f"\nEquation:")
print(f"Price = {size_m:.2f}×size + {bedroom_m:.2f}×bedrooms + {age_m:.2f}×age + {intercept:.2f}")

# ── TASK 4 — EVALUATE ────────────────────────────────────────────────────────
predictions = model.predict(X_test)

mae      = mean_absolute_error(y_test, predictions)
mse      = mean_squared_error(y_test, predictions)
rmse     = np.sqrt(mse)
r_square = r2_score(y_test, predictions)

print("\n" + "=" * 50)
print("           MODEL EVALUATION")
print("=" * 50)
print(f"MAE:      ${mae:,.2f}")
print(f"MSE:      ${mse:,.2f}")
print(f"RMSE:     ${rmse:,.2f}")
print(f"R² Score: {r_square:.2f}")
print("\nActual vs Predicted:")
for actual, predicted in zip(y_test, predictions):
    print(f"  Actual: ${actual:,.0f}  →  Predicted: ${predicted:,.0f}")

# ── TASK 5 — NEW HOUSE PREDICTIONS ───────────────────────────────────────────
new_houses = [
    [1800, 3, 5],
    [2500, 4, 2],
    [1000, 2, 15],
]

new_df           = pd.DataFrame(new_houses, columns=["Size", "Bedrooms", "Age"])
new_predictions  = model.predict(new_houses)
new_df["Predicted Price"] = new_predictions

print("\n" + "=" * 50)
print("         NEW HOUSE PREDICTIONS")
print("=" * 50)
print(new_df.to_string(index=False))

# Bonus — 5000 sqft, 6 bed, 1 year old
bonus_house = [[5000, 6, 1]]
bonus_pred  = model.predict(bonus_house)
print(f"\nBonus → 5000sqft, 6bed, 1yr old: ${bonus_pred[0]:,.0f}")

# ── TASK 6 — VISUALIZE ───────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
fig.set_facecolor("#F8FAFC")
fig.suptitle("House Price Predictor — Multiple Linear Regression",
             fontsize=16, fontweight="bold")

# Left — Size vs Price scatter
ax[0].scatter(size, prices,
              edgecolor="black", color="steelblue",
              s=100, zorder=3)
ax[0].set_title("Size vs Price", fontsize=13, fontweight="bold")
ax[0].set_xlabel("Size (sqft)", fontsize=12, fontweight="bold")
ax[0].set_ylabel("Price ($)", fontsize=12, fontweight="bold")
ax[0].set_facecolor("#FFFFFF")
ax[0].grid(linestyle="--", color="gray", alpha=0.5, axis="both")
ax[0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

# Right — Actual vs Predicted
ax[1].scatter(y_test, predictions,
              edgecolor="black", color="blue",
              s=100, label="Predictions", zorder=3)

#Perfect prediction diagonal line
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())
ax[1].plot([min_val, max_val],
           [min_val, max_val],
           color="red", linestyle="--",
           linewidth=2, label="Perfect Prediction")

# m,b = np.polyfit(y_test,predictions,1)

# y = [m*x + b for x in y_test]

# ax[1].plot(y_test,y, color="red", linestyle="--",
#      linewidth=2, label="Perfect Prediction")

# Metrics box
metric_text = f"MAE:  ${mae:,.0f}\nRMSE: ${rmse:,.0f}\nR²:   {r_square:.2f}"
ax[1].text(0.02, 0.97, metric_text,
           transform=ax[1].transAxes,
           fontsize=11,
           verticalalignment="top",
           fontfamily="monospace",
           bbox=dict(boxstyle="round",
                     facecolor="#E2E8F0",
                     alpha=0.8),
           zorder=5)

ax[1].set_title("Actual vs Predicted Prices",
                fontsize=13, fontweight="bold")
ax[1].set_xlabel("Actual Price ($)", fontsize=12, fontweight="bold")
ax[1].set_ylabel("Predicted Price ($)", fontsize=12, fontweight="bold")
ax[1].legend(loc="lower right", fontsize=11)
ax[1].set_facecolor("#FFFFFF")
ax[1].grid(linestyle="--", color="gray", alpha=0.5, axis="both")
ax[1].xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax[1].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

plt.tight_layout()
plt.savefig("house_price_predictor.png", dpi=150, bbox_inches="tight")
plt.show()


