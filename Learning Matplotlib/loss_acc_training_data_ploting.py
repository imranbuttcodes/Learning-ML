import matplotlib.pyplot as plt
import numpy as np

epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

train_loss = [0.95, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15]
val_loss   = [0.98, 0.85, 0.75, 0.65, 0.55, 0.50, 0.55, 0.62, 0.70, 0.80]

train_acc  = [45, 55, 63, 70, 76, 82, 87, 91, 93, 96]
val_acc    = [42, 52, 60, 67, 73, 78, 75, 70, 65, 60]

# Confusion matrix after training
confusion = [[48,  4,  8],
             [ 5, 42,  3],
             [ 7,  3, 40]]

classes = ["Car", "Bike", "Truck"]

fig, ax = plt.subplots(1,3,figsize=(18,5))

ax[0].grid(
    linestyle='--',
    color = 'gray',
    alpha = 0.7,
    axis = 'both'
)

ax[0].plot(epochs, train_loss,color='blue',linestyle='-',marker='o',linewidth=2,label='Train loss')

ax[0].plot(epochs, val_loss,color='red',linestyle='--',marker='s',linewidth=2,label='Val loss')

ax[0].legend(
    loc ='best',
    title = 'Loss',
    fontsize = 12,
)

ax[0].set_title("Loss Analysis", fontsize = 15, fontweight= 'bold', fontstyle = 'italic', color='green')

ax[0].set_xlabel('Epochs', color = 'r', fontweight='bold')

ax[0].set_ylabel('loss', color='r', fontweight = 'bold')

ax[0].annotate(
    'Overfitting starts here',
    xy=(6,0.50),
    xytext=(6,0.8),
    arrowprops = {'arrowstyle': '->', 'color': 'red'},
    fontsize = 10,
    color = 'r',
    fontweight = 'bold',
)

ax[1].grid(
    linestyle='--',
    color = 'gray',
    alpha = 0.7,
    axis = 'both'
)



ax[1].plot(epochs, train_acc,color='blue',linestyle='-',marker='o',linewidth=2,label='Train loss')

ax[1].plot(epochs, val_acc,color='red',linestyle='--',marker='s',linewidth=2,label='Val loss')

ax[1].legend(
    loc = 'best',
    title = 'Loss',
    fontsize = 12,
    title_fontproperties = {'weight': 'bold'}
)


ax[1].set_title("Model Accuracy Analysis", fontsize = 15, fontweight= 'bold', fontstyle = 'italic', color='green')

ax[1].set_xlabel('Epochs', color = 'r', fontweight='bold')

ax[1].set_ylabel('Accuracy', color='r', fontweight = 'bold')



ax[1].annotate(
    'Overfitting starts here',
    xy=(6,78),
    xytext=(6,50),
    arrowprops = {'arrowstyle': '->', 'color': 'red'},
    fontsize = 10,
    color = 'r',
    fontweight = 'bold',
)

img = ax[2].imshow(confusion,cmap='Blues')

plt.colorbar(img)


for i in range(3):
    for j in range(3):
        ax[2].text(
            j,
            i,
            confusion[i][j],
            ha = 'center',
            va = 'center',
            fontsize = 12,
            color = 'white' if confusion[i][j] > 39 else 'black',
        )
ax[2].set_xlabel('Predicted', fontweight = 'bold', fontsize= 12)
ax[2].set_ylabel('Actual', fontweight = 'bold', fontsize = 12)

# ax[2].set_xlim(0,2)
# ax[2].set_ylim(0,2)
ax[2].set_xticks(np.arange(0,3,1))
ax[2].set_yticks(np.arange(0,3,1))

ax[2].set_xticklabels(classes)
ax[2].set_yticklabels(classes)

ax[2].set_title('Confusion Matrix of model', fontsize = 15, fontweight = 'bold', color='green', fontstyle = 'italic')

fig.suptitle('Vehicle Classifier Training Report', color = 'black', fontsize = 19, fontweight = 'bold')

fig.savefig(
    'training_report.png',
    dpi = 150,
    bbox_inches = 'tight',
    pad_inches = 0.1,
)


plt.tight_layout()

plt.show()

