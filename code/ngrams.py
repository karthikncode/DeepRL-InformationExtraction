''' script to find context words for all the tags'''

import sys, pdb, collections
from operator import itemgetter
from nltk.corpus import stopwords
import pdb

CONTEXT = 5

SENTENCE_CONTEXT = True

contextWords = collections.defaultdict(lambda:collections.defaultdict(lambda:0.))
stop = stopwords.words('english')

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
                    if word.isalpha() and not word in stop:
                        contextWords[tag][word] += 1

# print stats
for tag, bag in contextWords.items():
    print tag
    s = sorted(bag.items(), key=itemgetter(1), reverse=True)
    print s[:50]
    print "Total: ", sum(q[1] for q in s) 


