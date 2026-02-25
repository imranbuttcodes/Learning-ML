import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ── RAW MESSY DATA ───────────────────────────────────────────
data = {
    "Age":        [20, np.nan, 22, 19, np.nan, 21, 23, 20, 999, 22],
    "City":       ["Lahore", "Karachi", "Islamabad", "Lahore", "Karachi",
                   "Lahore", np.nan, "Karachi", "Islamabad", np.nan],
    "Grade":      ["A", "B", np.nan, "C", "B", "A", "B", np.nan, "C", "A"],
    "Attendance": [90, 85, 78, np.nan, 92, 88, np.nan, 95, 82, 89],
    "Score":      [88, 76, 82, 91, np.nan, 85, 79, np.nan, 83, 87],
    "Passed":     [1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

print("=" * 50)
print("        STEP 1 — RAW DATA")
print("=" * 50)
print(df)
print(f"\nShape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ── STEP 2 — HANDLE OUTLIERS ─────────────────────────────────
print("\n" + "=" * 50)
print("        STEP 2 — HANDLE OUTLIERS")
print("=" * 50)

Q1  = df["Age"].quantile(0.25)
Q3  = df["Age"].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR

print(f"Age boundaries: {lower:.1f} to {upper:.1f}")
print(f"Outliers found: {len(df[df['Age'] > upper])}")

df["Age"] = df["Age"].clip(lower=lower, upper=upper)
print("Outliers handled! ✅")

# ── STEP 3 — HANDLE MISSING VALUES ───────────────────────────
print("\n" + "=" * 50)
print("        STEP 3 — HANDLE MISSING VALUES")
print("=" * 50)

df = df.fillna({
    "Age":        df["Age"].mean(),
    "City":       df["City"].mode()[0],
    "Grade":      df["Grade"].mode()[0],
    "Attendance": df["Attendance"].median(),
    "Score":      df["Score"].mean()
})

print("Missing values handled! ✅")
print(f"Missing values remaining:\n{df.isnull().sum()}")

# ── STEP 4 — ENCODE CATEGORICAL DATA ─────────────────────────
print("\n" + "=" * 50)
print("        STEP 4 — ENCODE CATEGORICAL DATA")
print("=" * 50)

le = LabelEncoder()
df["Grade"] = le.fit_transform(df["Grade"])
df = pd.get_dummies(df, columns=["City"], dtype=int)

print("Encoding done! ✅")
print(f"Columns: {list(df.columns)}")

# ── STEP 5 — FEATURE SCALING ──────────────────────────────────
print("\n" + "=" * 50)
print("        STEP 5 — FEATURE SCALING")
print("=" * 50)

scaler = StandardScaler()
cols_to_scale = ["Age", "Attendance", "Score"]
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("Scaling done! ✅")

# ── STEP 6 — FINAL CLEAN DATASET ─────────────────────────────
print("\n" + "=" * 50)
print("        STEP 6 — FINAL CLEAN DATASET")
print("=" * 50)

print(np.round(df, 2))
print(f"\nFinal shape: {df.shape}")

# ── STEP 7 — TRAIN MODEL ON CLEAN DATA ───────────────────────
print("\n" + "=" * 50)
print("        STEP 7 — TRAIN MODEL")
print("=" * 50)

X = df.drop("Passed", axis=1).values
y = df["Passed"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions)*100:.1f}%")
print("Model trained on clean preprocessed data! ✅")

## What This Pipeline Does

# Step 1 → Load raw messy data
# Step 2 → Fix outliers first!
# Step 3 → Fill missing values
# Step 4 → Encode categories
# Step 5 → Scale features
# Step 6 → Clean data ready!
# Step 7 → Train model!
# ```

# ---

# ## Why This Order Matters
# ```
# Outliers BEFORE missing values
# → outlier might affect mean/median
# → fix outliers first then calculate mean!

# Missing values BEFORE encoding
# → can't encode NaN values
# → fill missing first!

# Encoding BEFORE scaling
# → need numbers before scaling
# → encode categories first!

# Scaling BEFORE training
# → model needs scaled data
# → always last step before training!