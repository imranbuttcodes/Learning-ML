import matplotlib.pyplot as plt


fig,ax = plt.subplots(figsize=(12,8))

ax.plot([1,2,3,4],[5,6,7,8],marker='o')

ax.plot([1,2,3,4],[7,9,11,15],marker='o')

ax.fill_between([1,2,3,4], [5,6,7,8], [7,9,11,15],
                 alpha=0.08, color="#2563EB")

plt.show()