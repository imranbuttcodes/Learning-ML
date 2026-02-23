import matplotlib.pyplot as plt

# print(plt.style.available)

# plt.style.use('classic')

# hours  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# scores = [35,45,67,78,89,32,112,12,34,56]

# plt.plot(scores,hours)

# plt.tight_layout()
# plt.show()



epochs     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_loss = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15]
val_loss   = [0.95, 0.75, 0.65, 0.55, 0.48, 0.45, 0.5, 0.58, 0.65, 0.75]

styles = ["seaborn-v0_8", "ggplot",
          "fivethirtyeight", "dark_background"]

fig, ax = plt.subplots(2, 2, figsize=(14, 8))
axes = ax.flatten()  # convert 2x2 grid into flat list

for i, style in enumerate(styles):
    with plt.style.context(style):
        axes[i].plot(epochs, train_loss, label="Train Loss")
        axes[i].plot(epochs, val_loss, label="Val Loss")
        axes[i].set_title(f"Style: {style}", fontweight="bold")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].legend()

plt.tight_layout()
plt.show()