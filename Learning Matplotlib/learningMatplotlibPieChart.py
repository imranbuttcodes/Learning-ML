import matplotlib.pyplot as plt

labels = ["Laptop", "Phone", "Tablet", "TV"]
sales  = [35, 22, 15, 24]
# fig, ax = plt.subplots(figsize=(12,8))
# ax.pie(sales,labels=labels)
# ax.set_title("Sales of Tech Products")

# ax.set_facecolor('green')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots()
# colors = ["#202457", "#E1B5CF", "#61777F", "#6A6E77"]
# explode = [0.0,0.0,0.1,0.0] # to pull third slice out by 0.1
# #startangle is used to rotate the pie chart by some degrees 
# ax.pie(sales,labels=labels,colors=colors,autopct='%1.2f%%',explode=explode, startangle=90)
# ax.set_title('Bro')

# plt.tight_layout()
# plt.show()




# labels  = ["Laptop", "Phone", "Tablet", "TV"]
# sales   = [35, 22, 15, 24]
# colors  = ["#2563EB", "#16A34A", "#DC2626", "#F59E0B"]
# explode = [0.1, 0, 0, 0]

# fig, ax = plt.subplots(figsize=(8, 6))

# ax.pie(sales,
#        labels=labels,
#        colors=colors,
#        explode=explode,
#        autopct="%1.1f%%",
#        startangle=90)

# ax.set_title("Sales by Product", fontsize=16, fontweight="bold")

# plt.tight_layout()
# plt.show()


# Dataset class distribution
labels  = ["Not Spam", "Spam"]
counts  = [950, 50]
colors  = ["#16A34A", "#DC2626"]
explode = [0, 0.1]   # highlight the minority class

fig, ax = plt.subplots(figsize=(8, 6))

ax.pie(counts,
       labels=labels,
       colors=colors,
       explode=explode,
       autopct="%1.1f%%",
       startangle=90)

ax.set_title("Dataset Class Balance",
             fontsize=16, fontweight="bold")

plt.tight_layout()
plt.show()

