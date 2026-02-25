import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

train_loss = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15]
val_loss   = [0.95, 0.75, 0.65, 0.55, 0.48, 0.45, 0.5, 0.58, 0.65, 0.75]

train_acc  = [55, 65, 70, 75, 80, 85, 87, 90, 91, 93]
val_acc    = [50, 62, 68, 73, 78, 80, 78, 75, 72, 68]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Left — Loss curves
ax[0].plot(epochs, train_loss, color="blue",
           marker="o", linewidth=2, label="Train Loss")
ax[0].plot(epochs, val_loss, color="red",
           marker="s", linewidth=2, label="Val Loss")
ax[0].set_title("Loss Curves", fontsize=14, fontweight="bold")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()
ax[0].grid(axis="both", linestyle="--", alpha=0.5)

# Right — Accuracy curves
ax[1].plot(epochs, train_acc, color="blue",
           marker="o", linewidth=2, label="Train Accuracy")
ax[1].plot(epochs, val_acc, color="red",
           marker="s", linewidth=2, label="Val Accuracy")
ax[1].set_title("Accuracy Curves", fontsize=14, fontweight="bold")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy %")
ax[1].legend()
ax[1].grid(axis="both", linestyle="--", alpha=0.5)

fig.suptitle("Model Training Report",
             fontsize=18, fontweight="bold")

plt.tight_layout()
plt.show()