''' plot R, Q, completion rate for multiple files at the same time'''

import sys, argparse
import matplotlib.pyplot as plt
import math
import numpy as np

# plt.gcf().subplots_adjust(bottom=0.15)


#rewards first
f1 = map(float, file(sys.argv[1]).read().split('\n')[1:-1])

g = file(sys.argv[2]).read().split('------------\n')
entity = int(sys.argv[3])
f2 = []
for ele in g[1:]:
    f2.append(float(ele.split('\n')[entity+1].split()[-1]))

max_epochs = 40
N = min(max_epochs, len(f1))

colors = ['red', 'orange', 'b']
markers = ['x', 6, '.']
# linestyles = ['-', '--', '-.', ':']

linestyles = ['-', '-','-']
labels = ['LSTM-DQN', 'BI-DQN', 'BOW-DQN']

fig, ax1 = plt.subplots()
ax1.plot(np.arange(N), f1[:N], 'b-')
ax1.set_xlabel('Epoch')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('Reward', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(np.arange(N), f2[:N], 'r--')
ax2.set_ylabel('Accuracy', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
# plt.show()
plt.savefig('plots/plot.pdf')





# for i in range(len(f)):
#     plt.plot(f[i][:N], color=colors[i], label=labels[i], linestyle=linestyles[i], markersize=6, linewidth=3) #normal scale
#     # plt.plot([-math.log(abs(x)) for x in f[i][:N]], color=colors[i], label=labels[i], linestyle=linestyles[i], markersize=6, linewidth=3) #log scale

# plt.xlabel('Epochs', fontsize=20)
# plt.ylabel('Reward', fontsize=25)

# plt.legend(loc=4, fontsize=15)
# labelSize=17
# plt.tick_params(axis='x', labelsize=labelSize)
# plt.tick_params(axis='y', labelsize=labelSize)


# x1,x2,y1,y2 = plt.axis()
# plt.axis((x1,x2,y1,1.2)) #set y axis limit



# plt.savefig('plots/plot.pdf')
