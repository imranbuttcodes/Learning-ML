import matplotlib.pyplot as plt
import numpy as np

scores = [32, 45, 55, 61, 63, 65, 68, 70, 71,
          72, 74, 75, 75, 78, 80, 82, 85, 88, 91, 95]

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0,100)
ax.set_xticks(np.arange(0,101,2))
ax.set_ylim(0,10)
ax.set_yticks(np.arange(0,11,1))
n,bins,bars = ax.hist(scores,bins=5,edgecolor='black')

ax.grid(
    color='gray',
    axis='both',
    linestyle='--'

)

ax.set_xlabel("Scores")
ax.set_ylabel("Total Scores")


for bar in bars:
    ax.text(
        bar.get_x()  + bar.get_width()/2,
        bar.get_height() + 0.5,
        int(bar.get_height()),
        ha = 'center',
        color = 'r',
    )



plt.tight_layout()
plt.show()

print('counts:',n)
print('bins:',bins)


# """
# One important tip — grid goes behind bars
# Always call ax.grid() before ax.hist() or ax.bar() so grid lines go behind the bars not on top:

# """
# scores = [32, 45, 55, 61, 63, 65, 68, 70, 71,
#           72, 74, 75, 75, 78, 80, 82, 85, 88, 91, 95]

# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# # Left — default ticks

# ax[0].grid(color = 'gray',
#            linestyle ='--',
#            axis='y',
#            alpha = 0.3,         #alpha = transparency ranges from 0 to 1
#            linewidth = 3) 
# ax[1].grid()

# ax[0].hist(scores, bins=5,color = "#b39595ff", edgecolor="black")
# ax[0].set_title("Default Ticks")
# # Right — custom ticks
# ax[1].hist(scores, bins=8, edgecolor="black",color = "#ff9494ff")
# ax[1].set_xlim(0, 100)
# ax[1].set_ylim(0, 10)
# ax[1].set_xticks(np.arange(0, 101, 10))
# ax[1].set_yticks(np.arange(0, 11, 1))
# ax[1].set_title("Custom Ticks")
# fig.set_facecolor("#EFE7E700")



# plt.tight_layout()
# plt.show()