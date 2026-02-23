import matplotlib.pyplot as plt
import numpy as np
# import numpy as np
# #Challange 1

# # plt.plot([1,2,3,4,5,6],[0.2,0.4,0.6,9.4,10.2,20.6]
# #          ,linestyle='--',
# #          color='r',
# #          marker='o')
# # plt.title('BRO TITLE')
# # plt.xlabel('X AXIS BRO')
# # plt.ylabel("Y AXIS BRO")
# # plt.show()


# # Challange 3

# # plt.plot([1,2,3,4,5,6],[1,1,1,1,1,100],
# #          linewidth=3,
# #          marker='*',
# #          color='#701506',
# #          )
# # plt.plot([1,2,3,4],[1,4,9,16],marker='^',linestyle='none',markersize=6)

# # plt.title('BRO TITLE')
# # plt.xlabel('X AXIS BRO')
# # plt.ylabel("Y AXIS BRO")
# # plt.show()

# #Grande challange 1
# #Data Given
# months = [1, 2, 3, 4, 5, 6]

# product_a = [15, 22, 18, 30, 27, 35]  # Laptop
# product_b = [10, 12, 20, 18, 25, 22]  # Phone
# product_c = [5,  8,  7,  12, 10, 15]  # Tablet

# plt.plot(months,product_a,
#          color="#2563EB",
#          linestyle='-',
#          marker='o')
# plt.plot(months,product_b,
#          color="#16A34A",
#          linestyle = '--',
#          marker='s',
#          )
# plt.plot(months,product_c,
#          color="#DC2626",
#          linestyle=':',
#          marker='^')
# #Challange 2
# plt.title("Product Sales Report - First Half 2024")
# plt.xlabel("Month")
# plt.ylabel("Unit sold")
# plt.show()
# #Challange 3
# months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
# fig, ax = plt.subplots()
# ax.set_facecolor("#16A34A")
# plt.plot(months,product_a,
#          color="#2563EB",
#          linestyle='-',
#          marker='o',
#          linewidth=3.5)
# plt.plot(months,product_b, 
#          color="#16A34A",
#          linestyle = '--',
#          marker='s',
#          )
# plt.plot(months,product_c,
#          color="#DC2626",
#          linestyle=':',
#          marker='^')


# plt.show()
# fig, ax = plt.subplots()
# ax.plot([1,2,3],[4,5,6])
# ax.set_xlabel('X axis')
# ax.set_ylabel("Y axis")
# ax.set_title("Title Bro")
# plt.show()




# fig,ax = plt.subplots(1,3)
# ax[0].plot([1,2,3],[4,5,6],marker='o',linestyle=':')
# ax[1].plot([1,2,3],[4,5,6],marker='^',color='r',linestyle='--')
# ax[2].plot([1,2,3],[4,5,6],marker='*',color='orange',linestyle='-.')

# ax[0].set_xlabel('Chart 1 x axis')
# ax[0].set_ylabel('Chart 1 Y axis')
# ax[0].set_title('Chart 1')

# ax[1].set_xlabel('Chart 2 x axis')
# ax[1].set_ylabel('Chart 2 Y axis')
# ax[1].set_title('Chart 2')

# ax[2].set_xlabel('Chart 3 x axis')
# ax[2].set_ylabel('Chart 3 Y axis')
# ax[2].set_title('Chart 3')
# plt.show()

# fig, ax = plt.subplots(3, 1)

# ax[0].plot([1,2,3], [1,4,9], color="blue")
# ax[0].set_title("Top Chart")

# ax[1].plot([1,2,3], [1,2,3], color="red")
# ax[1].set_title("Middle Chart")

# ax[2].plot([1,2,3], [9,4,1], color="green")
# ax[2].set_title("Bottom Chart")

# plt.show()

# Gride plots 
# fig, ax = plt.subplots(2, 2)

# ax[0, 0].plot([1,2,3], [1,4,9], color="blue")
# ax[0, 0].set_title("Top Left")

# ax[0, 1].plot([1,2,3], [9,4,1], color="red")
# ax[0, 1].set_title("Top Right")

# ax[1, 0].plot([1,2,3], [2,5,8], color="green")
# ax[1, 0].set_title("Bottom Left")

# ax[1, 1].plot([1,2,3], [8,5,2], color="orange")
# ax[1, 1].set_title("Bottom Right")

# plt.show()


# fig, ax = plt.subplots(2, 3, figsize=(8, 6))

# ax[0, 0].set_title("Top Left")
# ax[0, 1].set_title("Top Right")
# ax[1, 0].set_title("Bottom Left")
# ax[1, 1].set_title("Bottom Right")
# plt.tight_layout()
# plt.show()  # looks messy!


# fig, ax = plt.subplots(2, 2, figsize=(10, 7))

# fig.suptitle("My Sales Dashboard", fontsize=16, fontweight="bold")

# ax[0, 0].set_title("Chart 1")
# ax[0, 1].set_title("Chart 2")
# ax[1, 0].set_title("Chart 3")
# ax[1, 1].set_title("Chart 4")

# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(figsize=(8,12))

# fig.set_facecolor("#1A1A2E")   # outer background (the frame)
# ax.set_facecolor("#BB8F3D")    # inner background (the chart area)
# ax.plot([1,2,3], [4,5,6], color="red")
# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# ax[0].plot([1,2,3], [4,5,6], color="blue")
# ax[0].set_title("Left Chart")

# ax[1].plot([1,2,3], [6,5,4], color="red")
# ax[1].set_title("Right Chart")
# plt.tight_layout()
# plt.show()


# fig, ax = plt.subplots(2, 2, figsize=(8, 6))

# fig.suptitle('SUPER TITLE',fontsize=26,fontweight='bold',fontstyle = 'italic')
# ax[0, 0].plot([1,2,3], [4,5,6], color="blue")
# ax[0, 0].set_title("Top Left",fontstyle='italic',fontsize=26,fontweight='bold')
# ax[0, 1].plot([1,2,3], [6,5,4], color="red")
# ax[0, 1].set_title("Top Right",fontstyle='italic',fontsize=26,fontweight='extra bold')

# ax[1, 0].plot([1,2,3], [1,4,9], color="green")
# ax[1, 0].set_title("Bottom Left",fontstyle='italic',fontsize=26,fontweight='light')

# ax[1, 1].plot([1,2,3], [9,4,1], color="orange")
# ax[1, 1].set_title("Bottom Right",fontstyle='italic',fontsize=26,fontweight='normal')
# ax[1,1].set_xlabel("BRO")

# plt.tight_layout()
# plt.show()


#List of common fonstsyles and weights

"""
Font Styles (fontstyle)
There are three standard string options for the slant of the text: 

    'normal': The default upright font.
    'italic': A cursive-like slanted version of the font.
    'oblique': A mechanically slanted version of the upright font (often looks similar to italic). 

Font Weights (fontweight)
Matplotlib supports both descriptive strings and numeric values (0–1000). 
String Options:

    'ultralight'
    'light'
    'normal' (equivalent to 'regular', 'book', or 'roman')
    'medium'
    'semibold' (equivalent to 'demibold' or 'demi')
    'bold'
    'heavy'
    'extra bold'
    'black' 

Numeric Mappings:
If you use integers, they typically map to standard weights as follows: 

    400: Normal
    500: Medium
    600: Semi Bold
    700: Bold
    900: Black / Heavy


"""




# fig, ax = plt.subplots(figsize=(8,5))
# ax.plot([1,2,3],[4,5,6],color="#000000ff")
# #Chart's background color
# ax.set_facecolor('r') # you can also pass html color code instead of str(r)
# #whole window's background color!
# fig.set_facecolor('g')
# plt.show()


#Practice Challange

# months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

# laptop_sales  = [15, 22, 18, 30, 27, 35]
# phone_sales   = [10, 12, 20, 18, 25, 22]
# tablet_sales  = [5,  8,  7,  12, 10, 15]
# tv_sales      = [8,  14, 11, 20, 17, 24]

# fig,ax = plt.subplots(2,2,figsize=(12,8))

# #plotting charts 1-4

# ax[0,0].plot(months,laptop_sales,color='b',marker='o',linestyle='-',markersize=8,linewidth=2,label='Laptop')
# ax[0,0].set_title('Laptop Sales', fontsize = 20, fontstyle='italic',fontweight='bold',color='blue')
# ax[0,0].set_xlabel('Month',color='b')
# ax[0,0].set_ylabel('Units Sold',color='b')

# ax[0,1].plot(months,phone_sales,color='g',marker='s',linestyle='--',markersize=8,linewidth=2,label='Phone')
# ax[0,1].set_title('Phone Sales', fontsize = 19, fontstyle='oblique',fontweight='heavy', color='g')
# ax[0,1].set_xlabel('Month', color='g')
# ax[0,1].set_ylabel('Units Sold', color='g')

# ax[1,0].plot(months,tablet_sales,color='r',marker='^',linestyle=':',markersize=8,linewidth=2,label='Tablet')
# ax[1,0].set_title('Tablet Sales', fontsize = 12, fontstyle='normal',fontweight='ultralight', color='r')
# ax[1,0].set_xlabel('Month', color='r')
# ax[1,0].set_ylabel('Units Sold', color='r')

# ax[1,1].plot(months,tv_sales,color='purple',marker='*',linestyle='-.',markersize=8,linewidth=2,label='TV')
# ax[1,1].set_title('TV Sales', fontsize = 19, fontstyle='italic',fontweight='light', color='purple')
# ax[1,1].set_xlabel('Month', color='purple')
# ax[1,1].set_ylabel('Units Sold', color='purple')

# fig.suptitle('Products Sales Analysis',fontweight='bold',fontsize=18)
# fig.set_facecolor("#c2c2c2ff")

# ax[0,0].set_facecolor("#BDABED")
# ax[0,1].set_facecolor("#BDABED")
# ax[1,0].set_facecolor("#BDABED")
# ax[1,1].set_facecolor("#BDABED")

# # default legend settings

# # ax[0,0].legend()
# # ax[0,1].legend()
# # ax[1,0].legend()
# # ax[1,1].legend()

# #Costumized legends settings

# ax[0,0].legend(loc='upper left',title='Product', title_fontproperties={'weight':'bold'}) # that's default location
# ax[0,1].legend(loc='lower right',title='Product', fontsize=12,facecolor="#B2148D00", title_fontproperties={'weight':'heavy'})
# ax[1,0].legend(loc='lower right', title='Product',fontsize=19,facecolor="#09EF0900")
# ax[1,1].legend(loc='best', title='Product',fontsize= 10,facecolor="#7CCE0800")
# # for loc there's also other settings like center, bottom etcc you can google it 
# plt.tight_layout()
# plt.show()

# Grids

# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot([1,2,3],[4,5,6],color='r')
# ax.grid(
#     linestyle='--',
#     color='gray',
#     linewidth=2,
#     axis='y',
# )
# plt.tight_layout()
# plt.show()



x = [0,1,2,3,4,5]

y = []

for i in x:
    y.append(2*i + 3)

fig, ax = plt.subplots(figsize = (12,8))

ax.plot(x,y,marker = 'o')
ax.set_title("Equation 2x + 3")

ax.set_xlabel("X axis")
ax.set_ylabel("Y Axis")
ax.grid(linestyle='--',
        color='gray',
        axis='both',
        alpha = 0.7,       #alpha = transparency ranges from 0 to 1 
        )
# Set boundaries from 0 to 15
ax.set_xlim(0, 15)
ax.set_ylim(0, 15)

# Set tick spacing (e.g., every 1 unit)
ax.set_xticks(np.arange(0, 16, 1))
ax.set_yticks(np.arange(0, 16, 1))

ax.annotate(
    'Highest Value',
    xy=(5,13),
    xytext=(4,8),
    arrowprops=dict(arrowstyle='->',color = 'black',linewidth = 2), #color of the line and width of the line
    color = 'red', # color of the text
    fontsize = 18,
    fontweight = 'bold'
)
plt.tight_layout()

plt.show()