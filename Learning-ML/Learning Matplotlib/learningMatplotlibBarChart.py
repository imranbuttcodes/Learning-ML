import matplotlib.pyplot as plt

# categories = ["Laptop", "Phone", "Tablet", "TV"]
# sales      = [35, 22, 15, 24]
# plt.bar(categories,sales)

# plt.title("Its bar chart of tech products")
# plt.xlabel('Products')
# plt.ylabel('Sales')
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(2,2,figsize=(12,8))

# ax[0,0].bar(categories,sales,label='LEGENDS HE BRO',color='g',edgecolor='black',width=0.4)
# ax[1,0].bar(categories,sales)
# ax[0,1].bar(categories,sales)
# ax[1,1].bar(categories,sales)
# ax[0,0].legend(loc='best',fontsize=12).get_title().set_weight('bold')
# plt.show()


# categories = ["Laptop", "Phone", "Tablet", "TV"]
# sales      = [35, 22, 15, 24]

# fig, ax = plt.subplots(figsize=(8, 5))

# bars = ax.bar(categories, sales,
#               color="steelblue",
#               edgecolor="black",
#               width=0.5)

# # Add value labels on top of each bar
# for bar in bars:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2,
#             height + 0.5,
#             str(height),
#             ha="center",
#             fontsize=11,
#             fontweight="bold")

# ax.set_title("Total Sales by Product")
# ax.set_xlabel("Product")
# ax.set_ylabel("Units Sold")

# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(figsize=(8, 5))

# ax.text(1, 5, "Hello!")       # places "Hello!" at position x=1, y=5
# ax.text(2, 10, "Bro!")        # places "Bro!" at position x=2, y=10
# ax.text(3, 7, "Matplotlib!")  # places "Matplotlib!" at position x=3, y=7

# ax.set_xlim(0, 5)   # just to see the chart
# ax.set_ylim(0, 15)

# plt.show()
#ax.plot([1,2,3],[4,5,6]) 
# ax.text(1,5,'Hello')
# ax.text(2,5,'Bro',color='r')
# ax.text(3,7,'Matplotlib')
# ax.set_xlim(0,5)
# ax.set_ylim(0,15)


# plt.show()



# categories = ["Laptop", "Phone", "Tablet", "TV"]
# sales      = [35, 22, 15, 24]



# fig, ax = plt.subplots(2,2,figsize=(12,8))

# bar1 = ax[0,0].bar(categories,sales,label='Laptop',color='g',edgecolor='black',width=0.4)
# bar2 = ax[1,0].bar(categories,sales,label="Phone",color="#074893",edgecolor='black')
# bar3 = ax[0,1].bar(categories,sales,label='Tablet',color="#7A6F49",edgecolor="#030303")
# bar4 = ax[1,1].bar(categories,sales,label='TV',color="#8D20C0",edgecolor="#FF8800")
# ax[0,0].legend(loc='best',fontsize=12,title='Product',facecolor="#F1E9F6").get_title().set_weight('bold')
# ax[0,1].legend(loc='best',fontsize=12,title='Product',facecolor="#F1E9F6").get_title().set_weight('bold')
# ax[1,0].legend(loc='best',fontsize=12,title='Product',facecolor="#F1E9F6").get_title().set_weight('bold')
# ax[1,1].legend(loc='best',fontsize=12,title='Product',facecolor="#F1E9F6").get_title().set_weight('bold')

# for bar in bar1:
#     ax[0,0].text(
#         bar.get_x() + bar.get_width()/2,
#         bar.get_height() + 0.5,
#         bar.get_height(),
#         fontsize=12,
#         fontweight= 'bold',
#         ha= 'center' 
#     )

# for bar in bar2:
#     ax[1,0].text(
#         bar.get_x() + bar.get_width()/2,
#         bar.get_height() + 0.5,
#         bar.get_height(),
#         fontsize=12,
#         fontweight= 'bold',
#         ha= 'center' 
#     )

# for bar in bar3:
#     ax[0,1].text(
#         bar.get_x() + bar.get_width()/2,
#         bar.get_height() + 0.5,
#         bar.get_height(),
#         fontsize=12,
#         fontweight= 'bold',
#         ha= 'right' 
#     )

# for bar in bar4:
#     ax[1,1].text(
#         bar.get_x() + bar.get_width()/2,
#         bar.get_height() + 0.5,
#         bar.get_height(),
#         fontsize=12,
#         fontweight= 'bold',
#         ha= 'left' 
#     )        

# plt.tight_layout()
# plt.show()


# categories = ["Laptop", "Phone", "Tablet", "TV"]
# sales      = [35, 22, 15, 24]

# fig, ax = plt.subplots(figsize=(8, 5))

# bars = ax.barh(categories, sales,
#         color="steelblue",
#         edgecolor="black",height=0.3)

# ax.set_title("Total Sales by Product")
# ax.set_xlabel("Units Sold")
# ax.set_ylabel("Product")
# for bar in bars:
#    ax.text (
#       bar.get_width() + 1.0,
#       bar.get_y() + bar.get_height()/2,
#       bar.get_width(),
#       va='center',
#       fontsize = 12,
#       fontweight = 'bold',
#       color = 'r'
#    )
# plt.tight_layout()
# plt.show()

