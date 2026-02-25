import matplotlib.pyplot as plt

# hours  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# scores = [35, 45, 50, 58, 65, 70, 75, 82, 88, 95]

# fig, ax = plt.subplots(figsize=(8, 5))

# ax.plot(hours, scores, color="steelblue",
#         marker="o", linewidth=2)

# ax.annotate(
#     "High Score!",           # text To display
#     xy=(9, 88),              # Where the arrow points
#     xytext=(8, 40),          # Where the text sits
#     arrowprops={'arrowstyle': 'wedge', 'color': 'red'}, #Style and color of the arrow
#     fontsize=12,
#     fontweight='bold',
#     color='red'
# )

# ax.set_title("Study Hours vs Scores")
# ax.set_xlabel("Hours")
# ax.set_ylabel("Score")

# plt.tight_layout()
# plt.show()


"""
## Step 2 — Customizing the Arrow

The arrow is controlled by `arrowprops=dict(...)`. Let's go through each option:

---

### `arrowstyle` — Shape of the arrow

```python
arrowprops=dict(arrowstyle="->")    # simple arrow
arrowprops=dict(arrowstyle="<->")   # double arrow
arrowprops=dict(arrowstyle="->", color="red")  # with color
```

Common styles:

| arrowstyle | Shape |
|---|---|
| `"->"` | Simple arrow → |
| `"<-"` | Reverse arrow ← |
| `"<->"` | Double arrow ↔ |
| `"-"` | Just a line |
| `"fancy"` | Thick fancy arrow |
| `"wedge"` | Wedge shape |

---

### `color` — Arrow color

```python
arrowprops=dict(arrowstyle="->", color="red")
```

---

### `lw` — Arrow line thickness

```python
arrowprops=dict(arrowstyle="->", color="red", lw=2)
```

---

### See them all together

```python
import matplotlib.pyplot as plt

hours  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = [35, 45, 50, 58, 65, 70, 75, 82, 88, 95]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(hours, scores, color="steelblue",
        marker="o", linewidth=2)

# Simple arrow
ax.annotate("Best Score!",
            xy=(10, 95),
            xytext=(7, 80),
            arrowprops=dict(arrowstyle="->",
                           color="red",
                           lw=2),
            fontsize=12,
            fontweight="bold",
            color="red")

# Fancy arrow
ax.annotate("Lowest Score",
            xy=(1, 35),
            xytext=(3, 45),
            arrowprops=dict(arrowstyle="fancy",
                           color="purple"),
            fontsize=12,
            fontweight="bold",
            color="purple")

ax.set_title("Study Hours vs Scores")
ax.set_xlabel("Hours")
ax.set_ylabel("Score")

plt.tight_layout()
plt.show()
```
---
"""


epochs      = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_loss  = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15]
val_loss    = [0.95, 0.75, 0.65, 0.55, 0.48, 0.45, 0.5, 0.58, 0.65, 0.75]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs, train_loss, color="blue",
        marker="o", linewidth=2, label="Training Loss")
ax.plot(epochs, val_loss, color="red",
        marker="s", linewidth=2, label="Validation Loss")

# Annotate where overfitting starts
ax.annotate("Overfitting starts here!",
            xy=(6, 0.45),
            xytext=(4, 0.6),
            arrowprops=dict(arrowstyle="->",
                           color="black",
                           lw=2),
            fontsize=11,
            fontweight="bold",
            color="black")

ax.set_title("Training vs Validation Loss",
             fontsize=16, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(axis="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()