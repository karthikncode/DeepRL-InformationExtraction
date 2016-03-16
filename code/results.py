''' script to get the results from log files as an average of last 5 values after 40 epochs'''

import sys, re, collections
import pdb

f = file(sys.argv[1]).read().split('------------\n')
entity = int(sys.argv[2])
k = int(sys.argv[3]) if len(sys.argv)>3 else 100
n = int(sys.argv[4]) if len(sys.argv) > 4 else 20

total = 0.
cnt = 0.
print "File: ", sys.argv[1]
for i in range(0, n):
    print f[k+i].split('\n')[entity+1]
    total += float(f[k+i].split('\n')[entity+1].split()[-1])
    cnt += 1

print "Avg. ", total/cnt
