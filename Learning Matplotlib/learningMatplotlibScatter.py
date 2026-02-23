import matplotlib.pyplot as plt
import numpy as np


# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [15, 22, 18, 30, 27, 35, 28, 40, 33, 45]

# fig, ax = plt.subplots(figsize=(8, 5))

# ax.scatter(x, y)

# ax.set_title("My First Scatter Plot")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")

# plt.tight_layout()
# plt.show()


# Hours studied vs exam score
# hours  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# scores = [35, 45, 50, 58, 65, 70, 75, 82, 88, 95]

# fig, ax = plt.subplots(figsize=(8, 5))

# ax.scatter(hours, scores,
#            color="steelblue",
#            edgecolor="black",
#            s=100)              # s = size of each dot

# ax.set_title("Study Hours vs Exam Score")
# ax.set_xlabel("Hours Studied")
# ax.set_ylabel("Exam Score")

# plt.tight_layout()
# plt.show()



# Student data — hours studied and exam score
# hours  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# scores = [35, 45, 50, 58, 65, 70, 75, 82, 88, 95]

# # Pass = 1, Fail = 0
# results = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# failed_hours = [h for h , r in zip(hours,results) if r == 0]
# failed_scores = [s for s,r in zip(scores,results) if r == 0]

# pass_hours = [h for h,r in zip(hours,results) if r == 1]
# pass_scores = [s for s,r in zip(scores,results) if r == 1]

# fig, ax = plt.subplots(figsize = (12,8))

# ax.scatter(
#     failed_hours,failed_scores,
#     s=100,
#     color = 'b',
#     edgecolor='black',
#     label = 'failed'
# )

# ax.scatter(
#     pass_hours,
#     pass_scores,
#     s=100,
#     color = 'g',
#     edgecolor= 'black',
#     label = "Pass"
# )
# ax.set_title("Study hours vs score")
# ax.set_xlabel("Hours Studied",fontstyle = 'italic', weight='bold',color = 'black')
# ax.set_ylabel("Exam Score")
# ax.legend(title = "Result",loc = 'upper left')
# plt.tight_layout()
# plt.show()



hours  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = [35, 45, 50, 58, 65, 70, 75, 82, 88, 95]

fig, ax = plt.subplots(figsize=(12,8))

m, b = np.polyfit(hours,scores,1)

y = [m*x + b for x in hours]

ax.scatter(hours,scores,
           s=100,
           color='r',
           edgecolor='black')
ax.plot(hours,y,color='black',label = 'Best Fit line')
ax.set_xlabel('hours studied',weight='bold',fontstyle='italic')
ax.set_ylabel('Scores')
ax.set_title("Analysis of hours studied and scores",fontsize=24,fontstyle='italic',fontweight='bold')
ax.legend(loc='best',fontsize=20,title='Predicted Line',facecolor = "#FBEEEE00").get_title().set_fontweight('bold')
plt.tight_layout()
plt.show()
