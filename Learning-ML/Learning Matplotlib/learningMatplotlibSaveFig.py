import matplotlib.pyplot as plt

epochs     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_loss = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15]

fig, ax = plt.subplots(figsize=(8, 5))

# ax.plot(epochs, train_loss, color="blue",
#         marker="o", linewidth=2)

# ax.set_title("Training Loss")
# ax.set_xlabel("Epoch")
# ax.set_ylabel("Loss")

# plt.tight_layout()
# plt.savefig("training_loss.png")   # saves before showing
# plt.show()

ax.plot(epochs,train_loss,color='b',marker='o',linewidth=2)

ax.set_title("Training Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")

plt.tight_layout()
# plt.savefig('chart3.png',dpi=2000,
#             bbox_inches='tight',
#             facecolor='white')

plt.savefig(
    "chart_.png",           # filename and format
    dpi=150,               # quality
    bbox_inches="tight",   # remove whitespace
    facecolor="white",     # background color
    edgecolor="none",      # edge color of figure
    transparent=False,     # transparent background
    pad_inches=0.1,        # padding around chart
)


"""

## `bbox_inches` — All Options

```python
plt.savefig("chart.png", bbox_inches="tight")   # removes extra whitespace
plt.savefig("chart.png", bbox_inches="standard") # keeps default whitespace
plt.savefig("chart.png", bbox_inches=None)       # same as standard
```

Honestly in real life you'll **always** use `bbox_inches="tight"`. It just makes saved charts look clean without ugly white borders. The other options exist but nobody really uses them.

---

## `dpi` — All You Need to Know

DPI stands for **Dots Per Inch** — basically the quality/sharpness of your saved image:

```python
plt.savefig("chart.png", dpi=72)    # low quality — small file size
plt.savefig("chart.png", dpi=100)   # default matplotlib
plt.savefig("chart.png", dpi=150)   # good quality — everyday use
plt.savefig("chart.png", dpi=300)   # high quality — reports and publishing
plt.savefig("chart.png", dpi=600)   # very high — professional printing
```

Rule of thumb for ML:
- Saving for yourself → `dpi=150`
- Saving for a report or paper → `dpi=300`

---

## All `savefig` Options Together

```python
plt.savefig(
    "chart.png",           # filename and format
    dpi=150,               # quality
    bbox_inches="tight",   # remove whitespace
    facecolor="white",     # background color
    edgecolor="none",      # edge color of figure
    transparent=False,     # transparent background
    pad_inches=0.1,        # padding around chart
)
```

### `transparent=True` — Transparent background

```python
plt.savefig("chart.png", transparent=True)
```

Saves the chart with no background — useful when putting charts inside presentations or websites where you want the background to show through.

---

### `pad_inches` — Padding around the chart

```python
plt.savefig("chart.png", bbox_inches="tight", pad_inches=0.1)  # small padding
plt.savefig("chart.png", bbox_inches="tight", pad_inches=0.5)  # more padding
```

Only works together with `bbox_inches="tight"`.

---

## Quick Reference

| Option | Recommended value | When to change |
|---|---|---|
| `dpi` | `150` | Use `300` for reports |
| `bbox_inches` | `"tight"` | Almost never |
| `facecolor` | `"white"` | If you want dark background |
| `transparent` | `False` | `True` for presentations |
| `pad_inches` | `0.1` | If chart feels too tight |

---

The combination you'll use 90% of the time in ML projects:

```python
plt.savefig("chart.png", dpi=150, bbox_inches="tight")
```

That's it. Simple and clean. 💪

---

"""
