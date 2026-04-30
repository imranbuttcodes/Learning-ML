# 📊 Matplotlib Cheat Sheet (Complete Single File) 🔥

Matplotlib is a Python library for creating data visualizations like line plots, bar charts, scatter plots, histograms, and more.

## 🚀 Installation
```bash
pip install matplotlib
```

**Import:**
```python
import matplotlib.pyplot as plt
```

---

## 📈 1. Line Plot
```python
x =[1][2][3][4]
y =[5]
plt.plot(x, y)
plt.show()
```

**Customization:**
```python
plt.plot(x, y, color='red', linestyle='--', linewidth=2, marker='o', markersize=8)
```

## 📊 2. Scatter Plot
```python
plt.scatter(x, y, color='blue', s=100)  # s=size
plt.show()
```

## 📉 3. Bar Plot
```python
categories = ['A', 'B', 'C']
values =[5]
plt.bar(categories, values, color='green', edgecolor='black')
plt.show()
```

## 📊 4. Histogram
```python
data =[2][3][4][1]
plt.hist(data, bins=5, color='purple', edgecolor='black')
plt.show()
```

## 🪟 5. Subplots
```python
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
# Access: ax, ax, ax, ax[1]
```

## 🏷️ 6. Titles & Labels
```python
plt.title("Graph Title")
plt.xlabel("X Axis") 
plt.ylabel("Y Axis")
```

## 📌 7. Legend
```python
plt.plot(x, y, label="Data")
plt.legend()
```

## 📉 8. Multiple Lines
```python
plt.plot(x, y1, label="Line 1")
plt.plot(x, y2, label="Line 2")
plt.legend()
```

## 🎯 9. Fill Between
```python
plt.fill_between(x, y1, y2, color="blue", alpha=0.3)
```

## 🧠 10. Annotation
```python
plt.annotate("Peak", xy=(3, 30), xytext=(2, 35),
             arrowprops=dict(arrowstyle="->"))
```

## 📏 11. Axis Limits
```python
plt.xlim(0, 10)
plt.ylim(0, 100)
```

## 🔢 12. Ticks
```python
plt.xticks()[3][4][2][1]
plt.yticks()[5]
```

## 🪟 13. Subplots Example
```python
fig, ax = plt.subplots(1, 2)
ax.plot(x, y); ax.set_title("Plot 1")
ax.bar(categories, values); ax.set_title("Plot 2")[1]
plt.tight_layout()
plt.show()
```

## 🧩 Key Functions
| Function | Use Case |
|----------|----------|
| `plot()` | Line graphs |
| `scatter()` | Points |
| `bar()` | Categorical |
| `hist()` | Distribution |
| `subplots()` | Multiple graphs |
| `fill_between()` | Shaded area |
| `annotate()` | Labels |

## 🚀 Pro Tips
- **Grid:** `plt.grid(True)`
- **Combine with:** NumPy + Pandas + ML models
- **Save:** `plt.savefig('plot.png', dpi=300)`

**Perfect for ML results, data analysis, and presentations! 🎯**