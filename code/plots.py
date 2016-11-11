''' plot R, Q, completion rate for multiple files at the same time'''

import sys, argparse
import matplotlib.pyplot as plt
import math
import numpy as np

# plt.gcf().subplots_adjust(bottom=0.15)
# plt.gcf().subplots_adjust(right=1.05)
# plt.gcf().subplots_adjust(left=-0.05)


#rewards first
f1 = [0.0] + map(float, file(sys.argv[1]).read().split('\n')[1:-1])

g = file(sys.argv[2]).read().split('------------\n')

max_epochs = 100
N = min(max_epochs, len(f1))

colors = ['r', 'm', 'b', 'g']
markers = ['x', 6, '.']
# linestyles = ['-', '--', '-.', ':']

linestyles = ['-', '-','-']
labels = ['LSTM-DQN', 'BI-DQN', 'BOW-DQN']

fig, ax1 = plt.subplots()
ax1.plot(np.arange(N), f1[:N], 'k-', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=20)
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('Reward', color='black', fontsize=20)
for tl in ax1.get_yticklabels():
    tl.set_color('black')
    tl.set_fontsize(17)

for tl in ax1.get_xticklabels():    
    tl.set_fontsize(17)

ax2 = ax1.twinx()
entity = int(sys.argv[3])
initial_values = [45.2, 69.7, 68.6, 53.7]
for entity in [0,1,2,3]:
# for entity in [entity]:
    f2 = [initial_values[entity]]
    for ele in g[1:]:
        f2.append(100.0 * float(ele.split('\n')[entity+1].split()[-1]))
   
    ax2.plot(np.arange(N), f2[:N], colors[entity]+'--', linewidth=2)

ax2.set_ylabel('Accuracy (%)', color='black', fontsize=20)
for tl in ax2.get_yticklabels():
    tl.set_color('black')
    tl.set_fontsize(17)
# plt.show()
plt.savefig('plots/plot_new.pdf', bbox_inches='tight')





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
