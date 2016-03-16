''' script to find context words for all the tags'''

import sys, pdb, collections
from operator import itemgetter
import pdb

CONTEXT = 5

SENTENCE_CONTEXT = True

contextWords = collections.defaultdict(lambda:collections.defaultdict(lambda:0.))

with open(sys.argv[1], 'r') as inFile:

    f = inFile.read().split('\n')
    for i in range(1, len(f), 2):
        tokens = f[i].split()
        for j, token in enumerate(tokens):
            if '_TAG' not in token:                
                tag = "_".join(token.split('_')[1:])
                if SENTENCE_CONTEXT:
                    contextList = []
                    k=1
                    while k<=10 and (j+k < len(tokens)) and tokens[j+k].split('_')[0] != '.':
                        contextList.append(tokens[j+k].split('_')[0])
                        k+=1
                    k=1
                    while k<=10 and (j-k >=0) and tokens[j-k].split('_')[0] != '.':
                        contextList.append(tokens[j-k].split('_')[0])
                        k+=1
                else:
                    contextList = [q.split('_')[0] for q in tokens[max(0, j-CONTEXT) : min(len(tokens), j+CONTEXT)]]
                for word in contextList:
                    word = word.lower()
                    if word.isalpha():
                        contextWords[tag][word] += 1

# print stats
for tag, bag in contextWords.items():
    print tag
    s = sorted(bag.items(), key=itemgetter(1), reverse=True)
    print s[:50]
    print "Total: ", sum(q[1] for q in s) 


# killedNum
# [('in', 246.0), ('dead', 182.0), ('shooting', 148.0), ('a', 128.0), ('and', 106.0), ('killed', 88.0), ('injured', 87.0), ('the', 84.0), ('people', 83.0), ('of', 73.0), ('were', 70.0), ('at', 70.0), ('three', 53.0), ('after', 50.0), ('others', 46.0), ('was', 42.0), ('police', 41.0), ('found', 41.0), ('shot', 38.0), ('that', 37.0), ('man', 36.0), ('on', 36.0), ('wounded', 34.0), ('to', 33.0), ('person', 33.0), ('four', 30.0), ('where', 30.0), ('left', 30.0), ('said', 27.0), ('victims', 27.0), ('thursday', 25.0), ('are', 25.0), ('home', 24.0), ('two', 23.0), ('one', 22.0), ('including', 20.0), ('county', 19.0), ('is', 19.0), ('early', 19.0), ('hurt', 18.0), ('party', 18.0), ('family', 17.0), ('gunman', 16.0), ('cypress', 16.0), ('say', 16.0), ('year', 15.0), ('killing', 15.0), ('murder', 14.0), ('park', 14.0), ('old', 14.0)]
# Total:  3944.0
# city
# [('in', 956.0), ('a', 762.0), ('the', 729.0), ('police', 596.0), ('shooting', 521.0), ('of', 410.0), ('at', 380.0), ('and', 332.0), ('to', 325.0), ('shot', 247.0), ('people', 237.0), ('were', 221.0), ('said', 201.0), ('on', 194.0), ('four', 172.0), ('dead', 155.0), ('after', 150.0), ('injured', 150.0), ('is', 135.0), ('department', 123.0), ('was', 116.0), ('by', 114.0), ('that', 111.0), ('three', 107.0), ('are', 105.0), ('wounded', 103.0), ('county', 99.0), ('killed', 99.0), ('with', 98.0), ('man', 96.0), ('one', 94.0), ('two', 91.0), ('year', 88.0), ('an', 87.0), ('for', 84.0), ('say', 83.0), ('early', 82.0), ('old', 79.0), ('home', 78.0), ('saturday', 74.0), ('city', 74.0), ('night', 71.0), ('according', 69.0), ('news', 68.0), ('outside', 66.0), ('cbs', 65.0), ('new', 64.0), ('from', 64.0), ('party', 64.0), ('morning', 62.0)]
# Total:  20123.0
# woundedNum
# [('in', 359.0), ('a', 253.0), ('shooting', 244.0), ('injured', 207.0), ('the', 196.0), ('were', 192.0), ('shot', 190.0), ('and', 181.0), ('people', 180.0), ('at', 165.0), ('wounded', 127.0), ('police', 108.0), ('of', 90.0), ('one', 88.0), ('dead', 84.0), ('to', 81.0), ('others', 79.0), ('killed', 73.0), ('after', 70.0), ('victims', 69.0), ('man', 55.0), ('party', 55.0), ('was', 49.0), ('that', 48.0), ('men', 47.0), ('on', 47.0), ('are', 44.0), ('year', 43.0), ('early', 43.0), ('other', 41.0), ('old', 40.0), ('by', 40.0), ('an', 36.0), ('four', 34.0), ('outside', 34.0), ('person', 33.0), ('say', 32.0), ('said', 32.0), ('three', 30.0), ('morning', 29.0), ('all', 28.0), ('drive', 28.0), ('with', 28.0), ('left', 27.0), ('hospital', 26.0), ('wounds', 26.0), ('leaves', 25.0), ('during', 24.0), ('cbs', 24.0), ('house', 22.0)]
# Total:  7112.0
# shooterName
# [('the', 389.0), ('a', 247.0), ('and', 244.0), ('of', 200.0), ('was', 157.0), ('to', 154.0), ('in', 144.0), ('as', 124.0), ('said', 120.0), ('his', 115.0), ('year', 108.0), ('old', 104.0), ('police', 92.0), ('at', 82.0), ('identified', 65.0), ('had', 59.0), ('on', 58.0), ('with', 57.0), ('that', 55.0), ('is', 46.0), ('were', 45.0), ('from', 45.0), ('have', 44.0), ('for', 42.0), ('shot', 42.0), ('who', 42.0), ('he', 41.0), ('jr', 38.0), ('after', 35.0), ('killed', 35.0), ('county', 35.0), ('shooting', 34.0), ('been', 32.0), ('dead', 32.0), ('three', 31.0), ('suspect', 31.0), ('has', 30.0), ('shooter', 30.0), ('say', 29.0), ('s', 29.0), ('her', 29.0), ('by', 29.0), ('home', 29.0), ('arrested', 28.0), ('charged', 28.0), ('wife', 28.0), ('before', 26.0), ('authorities', 26.0), ('man', 26.0), ('when', 25.0)]
# Total:  8308.0
