import matplotlib.pyplot as plt
import numpy as np

# # Confusion matrix data
# data = [[45, 5],
#         [8, 42]]

# fig, ax = plt.subplots(figsize=(6, 5))

# im = ax.imshow(data, cmap="Blues")

# ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")

# plt.colorbar(im)   # adds the color scale bar on the side

# plt.tight_layout()
# plt.show()

# data = np.array([[83,20],
#                  [40,100]])

# fig,ax = plt.subplots(figsize=(6,5))

# im = ax.imshow(data,cmap='Blues')

# plt.colorbar(im)

# plt.tight_layout()

# plt.show()




# 3x3 confusion matrix
# data = [[45,  3,  2],
#         [ 4, 38,  8],
#         [ 1,  5, 44]]

# fig, ax = plt.subplots(figsize=(6, 5))

# im = ax.imshow(data, cmap="Blues")
# plt.colorbar(im)

# ax.set_title("3 Class Confusion Matrix",
#              fontsize=16, fontweight="bold")

# plt.tight_layout()
# plt.show()


data = [[45,  3,  2],
        [ 4, 38,  8],
        [ 1,  5, 44]]

classes = ["Cat", "Dog", "Bird"]

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(data, cmap="Blues")
plt.colorbar(im)

# Add class names on x and y axis
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted", fontweight="bold", fontsize=12)
ax.set_ylabel("Actual", fontweight="bold", fontsize=12)
ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")

# Add numbers inside each box
for i in range(3):          # loop through rows
    for j in range(3):      # loop through columns
        ax.text(j, i,       # x=column, y=row
                data[i][j], # the number to show
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if data[i][j] > 20 else "black")

plt.tight_layout()
plt.show()