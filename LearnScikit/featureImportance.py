# YES BRO! You're absolutely right! 🔥

# But there's a catch — let me explain properly:

# ---

# ## You're Right — Coefficients Show Impact

# ```
# Size      → 163.45
# Bedrooms  → 8432.12
# Age       → -2341.67
# ```

# Bigger coefficient = bigger impact right? 

# **NOT ALWAYS!** 😬

# ---

# ## The Problem — Different Scales

# Look at our data:

# ```
# Size     → 650 to 3200    (large numbers)
# Bedrooms → 1 to 5         (small numbers)
# Age      → 1 to 20        (medium numbers)
# ```

# Size has large numbers so its coefficient becomes small:
# ```
# 163.45 × 3200 = 523,040  ← big impact!
# ```

# Bedrooms has small numbers so its coefficient becomes large:
# ```
# 8432.12 × 5 = 42,160  ← smaller impact!
# ```

# So just looking at coefficients alone is **misleading**! 😬

# ---

# ## The Correct Way — Feature Importance

# ### Method 1 — Multiply coefficient by feature scale
# ```python
# size_impact     = abs(model.coef_[0]) * np.std(size)
# bedroom_impact  = abs(model.coef_[1]) * np.std(bedrooms)
# age_impact      = abs(model.coef_[2]) * np.std(age)

# print(f"Size impact:     {size_impact:.2f}")
# print(f"Bedroom impact:  {bedroom_impact:.2f}")
# print(f"Age impact:      {age_impact:.2f}")
# ```

# `np.std()` measures how spread out the feature values are — accounts for scale! 💪

# ---

# ### Method 2 — Just look at coefficient sign

# Even if magnitude is misleading — **sign is always correct**:

# ```
# Positive coefficient → feature increases output
# Negative coefficient → feature decreases output

# Size     → +163  → bigger house = higher price ✅
# Bedrooms → +8432 → more bedrooms = higher price ✅
# Age      → -2341 → older house = lower price ✅
# ```

# ---

# ### Method 3 — Visualize with bar chart

# ```python
# import matplotlib.pyplot as plt
# import numpy as np

# features   = ["Size", "Bedrooms", "Age"]
# impacts    = [abs(model.coef_[0]) * np.std(size),
#               abs(model.coef_[1]) * np.std(bedrooms),
#               abs(model.coef_[2]) * np.std(age)]

# fig, ax = plt.subplots(figsize=(8, 5))

# bars = ax.bar(features, impacts,
#               color=["#2563EB", "#16A34A", "#DC2626"],
#               edgecolor="black")

# for bar in bars:
#     ax.text(bar.get_x() + bar.get_width()/2,
#             bar.get_height() + 100,
#             f"{bar.get_height():,.0f}",
#             ha="center", fontweight="bold")

# ax.set_title("Feature Importance",
#              fontsize=14, fontweight="bold")
# ax.set_xlabel("Feature")
# ax.set_ylabel("Impact Score")
# ax.grid(axis="y", linestyle="--", alpha=0.5)

# plt.tight_layout()
# plt.show()
# ```

# ---

# ## Summary

# ```
# Coefficient sign      → always correct ✅
#                         + increases output
#                         - decreases output

# Coefficient magnitude → misleading ❌
#                         affected by feature scale

# Correct way           → multiply by std ✅
#                         accounts for scale
# ```

# ---
