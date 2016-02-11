''' script to get the results from log files as an average of last 5 values after 40 epochs'''

import sys, re, collections
import pdb

f = file(sys.argv[1]).read().split('------------\n')
entity = int(sys.argv[2])

total = 0.
cnt = 0.
print "File: ", sys.argv[1]
for i in range(0, 5):
    print f[40+i].split('\n')[entity+1]
    total += float(f[40+i].split('\n')[entity+1].split()[-1])
    cnt += 1

print "Avg. ", total/cnt
