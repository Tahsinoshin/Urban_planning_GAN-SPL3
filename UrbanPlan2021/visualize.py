import numpy as np
import pickle
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import axes3d
from matplotlib.backends.backend_pdf import PdfPages

poi = np.load('./data/100_poi_dis.npz', encoding='latin1')
poi = poi['arr_0'][-299:]
# poi = np.load('./model/tmp/cluvae_generate_result_50.npz', encoding='latin1')
poi_data = poi.reshape(299,1,100,100,20)
poi_data = poi_data

n = 9
data0 = poi_data[n][0][:, :, 0]
data1 = poi_data[n][0][:, :, 1]
data2 = poi_data[n][0][:, :, 2]
data3 = poi_data[n][0][:, :, 3]
data4 = poi_data[n][0][:, :, 4]
data5 = poi_data[n][0][:, :, 5]
data6 = poi_data[n][0][:, :, 6]
data7 = poi_data[n][0][:, :, 7]
data8 = poi_data[n][0][:, :, 8]
data9 = poi_data[n][0][:, :, 9]
data10 = poi_data[n][0][:, :, 10]
data11 = poi_data[n][0][:, :, 11]
data12 = poi_data[n][0][:, :, 12]
data13 = poi_data[n][0][:, :, 13]
data14 = poi_data[n][0][:, :, 14]
data15 = poi_data[n][0][:, :, 15]
data16 = poi_data[n][0][:, :, 16]
data17 = poi_data[n][0][:, :, 17]
data18 = poi_data[n][0][:, :, 18]
data19 = poi_data[n][0][:, :, 19]
# data0 = poi['arr_0'][n][0, :, :]
# data1 = poi['arr_0'][n][1, :, :]
# data2 = poi['arr_0'][n][2, :, :]
# data3 = poi['arr_0'][n][3, :, :]
# data4 = poi['arr_0'][n][4, :, :]
# data5 = poi['arr_0'][n][5, :, :]
# data6 = poi['arr_0'][n][6, :, :]
# data7 = poi['arr_0'][n][7, :, :]
# data8 = poi['arr_0'][n][8, :, :]
# data9 = poi['arr_0'][n][9, :, :]
# data10 = poi['arr_0'][n][10, :, :]
# data11 = poi['arr_0'][n][11, :, :]
# data12 = poi['arr_0'][n][12, :, :]
# data13 = poi['arr_0'][n][13, :, :]
# data14 = poi['arr_0'][n][14, :, :]
# data15 = poi['arr_0'][n][15, :, :]
# data16 = poi['arr_0'][n][16, :, :]
# data17 = poi['arr_0'][n][17, :, :]
# data18 = poi['arr_0'][n][18, :, :]
# data19 = poi['arr_0'][n][19, :, :]

# set the grid parameter
def gen_pos(data):
    grid = 100
    x = []
    y = []
    z = []
    i = 0
    while i<grid:
        j = 0
        while j<grid:
            if data[i][j] > 0.0:
                x.append(i)
                y.append(j)
                z.append(data[i][j])
            j += 1
        i += 1
    return x, y, z

x0, y0, z0 = gen_pos(data0)
x1, y1, z1 = gen_pos(data1)
x2, y2, z2 = gen_pos(data2)
x3, y3, z3 = gen_pos(data3)
x4, y4, z4 = gen_pos(data4)
x5, y5, z5 = gen_pos(data5)
x6, y6, z6 = gen_pos(data6)
x7, y7, z7 = gen_pos(data7)
x8, y8, z8 = gen_pos(data8)
x9, y9, z9 = gen_pos(data9)
x10, y10, z10 = gen_pos(data10)
x11, y11, z11 = gen_pos(data11)
x12, y12, z12 = gen_pos(data12)
x13, y13, z13 = gen_pos(data13)
x14, y14, z14 = gen_pos(data14)
x15, y15, z15 = gen_pos(data15)
x16, y16, z16 = gen_pos(data16)
x17, y17, z17 = gen_pos(data17)
x18, y18, z18 = gen_pos(data18)
x19, y19, z19 = gen_pos(data19)

fig = plt.figure()
ax = pl.subplot(projection='3d')

# color_codes = ['b', 'g', 'r', 'c', 'm', 'y', 'blue', 'orange', 'green', 'coral', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'peru', 'tan', 'skyblue', 'orchid']
color_codes = ['gray', 'purple', 'orchid', 'plum', 'peru', 'orange', 'sandybrown', 'gold', 'darkkhaki', 'khaki', 'palegoldenrod', 'salmon', 'darkcyan', 'lightseagreen', 'darkturquoise', 'blue', 'royalblue', 'cornflowerblue', 'deepskyblue', 'skyblue']
# x_pos = x1
# y_pos = y1
# z_pos = np.zeros(len(z1))

# x_size = np.ones(len(x1))
# y_size = np.ones(len(y1))
# z_height = z1

trans = 1
try:
    ax.bar3d(x0, y0, np.zeros(len(z0)), np.ones(len(x0)), np.ones(len(y0)), z0, color=color_codes[0], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    ax.bar3d(x1, y1, np.zeros(len(z1)), np.ones(len(x1)), np.ones(len(y1)), z1, color=color_codes[1], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel2 = ax.bar3d(x2, y2, np.zeros(len(z2)), np.ones(len(x2)), np.ones(len(y2)), z2, color=color_codes[2], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel3 = ax.bar3d(x3, y3, np.zeros(len(z3)), np.ones(len(x3)), np.ones(len(y3)), z3, color=color_codes[3], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel4 = ax.bar3d(x4, y4, np.zeros(len(z4)), np.ones(len(x4)), np.ones(len(y4)), z4, color=color_codes[4], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel5 = ax.bar3d(x5, y5, np.zeros(len(z5)), np.ones(len(x5)), np.ones(len(y5)), z5, color=color_codes[5], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel6 = ax.bar3d(x6, y6, np.zeros(len(z6)), np.ones(len(x6)), np.ones(len(y6)), z6, color=color_codes[6], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel7 = ax.bar3d(x7, y7, np.zeros(len(z7)), np.ones(len(x7)), np.ones(len(y7)), z7, color=color_codes[7], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel8 = ax.bar3d(x8, y8, np.zeros(len(z8)), np.ones(len(x8)), np.ones(len(y8)), z8, color=color_codes[8], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel9 = ax.bar3d(x9, y9, np.zeros(len(z9)), np.ones(len(x9)), np.ones(len(y9)), z9, color=color_codes[9], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel10 = ax.bar3d(x10, y10, np.zeros(len(z10)), np.ones(len(x10)), np.ones(len(y10)), z10, color=color_codes[10], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel11 = ax.bar3d(x11, y11, np.zeros(len(z11)), np.ones(len(x11)), np.ones(len(y11)), z11, color=color_codes[11], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel12 = ax.bar3d(x12, y12, np.zeros(len(z12)), np.ones(len(x12)), np.ones(len(y12)), z12, color=color_codes[12], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel13 = ax.bar3d(x13, y13, np.zeros(len(z13)), np.ones(len(x13)), np.ones(len(y13)), z13, color=color_codes[13], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel14 = ax.bar3d(x14, y14, np.zeros(len(z14)), np.ones(len(x14)), np.ones(len(y14)), z14, color=color_codes[14], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel15 = ax.bar3d(x15, y15, np.zeros(len(z15)), np.ones(len(x15)), np.ones(len(y15)), z15, color=color_codes[15], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel16 = ax.bar3d(x16, y16, np.zeros(len(z16)), np.ones(len(x16)), np.ones(len(y16)), z16, color=color_codes[16], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel17 = ax.bar3d(x17, y17, np.zeros(len(z17)), np.ones(len(x17)), np.ones(len(y17)), z17, color=color_codes[17], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    channel18 = ax.bar3d(x18, y18, np.zeros(len(z18)), np.ones(len(x18)), np.ones(len(y18)), z18, color=color_codes[18], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
try:
    ax.bar3d(x19, y19, np.zeros(len(z19)), np.ones(len(x19)), np.ones(len(y19)), z19, color=color_codes[19], alpha=trans)
except ValueError:  #raised if `x, y, z` is empty.
    pass
channel0 = mpatches.Patch(color=color_codes[0], label='0')
channel1 = mpatches.Patch(color=color_codes[1], label='1')
channel2 = mpatches.Patch(color=color_codes[2], label='2')
channel3 = mpatches.Patch(color=color_codes[3], label='3')
channel4 = mpatches.Patch(color=color_codes[4], label='4')
channel5 = mpatches.Patch(color=color_codes[5], label='5')
channel6 = mpatches.Patch(color=color_codes[6], label='6')
channel7 = mpatches.Patch(color=color_codes[7], label='7')
channel8 = mpatches.Patch(color=color_codes[8], label='8')
channel9 = mpatches.Patch(color=color_codes[9], label='9')
channel10 = mpatches.Patch(color=color_codes[10], label='10')
channel11 = mpatches.Patch(color=color_codes[11], label='11')
channel12 = mpatches.Patch(color=color_codes[12], label='12')
channel13 = mpatches.Patch(color=color_codes[13], label='13')
channel14 = mpatches.Patch(color=color_codes[14], label='14')
channel15 = mpatches.Patch(color=color_codes[15], label='15')
channel16 = mpatches.Patch(color=color_codes[16], label='16')
channel17 = mpatches.Patch(color=color_codes[17], label='17')
channel18 = mpatches.Patch(color=color_codes[18], label='18')
channel19 = mpatches.Patch(color=color_codes[19], label='19')
plt.legend(handles=[channel0, channel1, channel2, channel3, channel4, channel5, channel6, channel7, channel8, channel9, channel10, channel11, channel12, channel13, channel14, channel15, channel16, channel17, channel18, channel19], loc='center left',bbox_to_anchor=(-0.1, 0.5), prop={'size': 8})
plt.savefig('original_9.pdf')
plt.show()

# [242, 211, 247,   1, 290, 163, 162, 190, 258, 270]

# array([  9,  68,   4, 111, 110,  13,  14,  84,  85,  74])