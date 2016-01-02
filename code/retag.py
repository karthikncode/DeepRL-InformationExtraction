''' script to retag the data files, especially for killed and wounded tags'''

import sys, pdb

CONTEXT = 5

cnt = 0
total = 0

with open(sys.argv[1], 'r') as inFile:
    with open(sys.argv[2], 'w') as outFile:
        f = inFile.read().split('\n')
        for i in range(1, len(f), 2):
            tokens = f[i].split()

            for j, token in enumerate(tokens):
                if 'killedNum' in token:
                    tmpStr = ' '.join([q.split('_')[0] for q in tokens[max(0, j-CONTEXT) : min(len(tokens), j+CONTEXT)]]).lower()
                    # print tmpStr
                    # pdb.set_trace()
                    if 'dead' not in tmpStr and 'kill' not in tmpStr and 'victim' not in tmpStr \
                    and 'fatal' not in tmpStr:
                        tokens[j] = token.replace('killedNum', 'TAG')
                        cnt += 1
                    total += 1
                elif 'woundedNum' in token:
                    tmpStr = ' '.join([q.split('_')[0] for q in tokens[max(0, j-CONTEXT) : min(len(tokens), j+CONTEXT)]]).lower()
                    if 'shot' not in tmpStr and 'injure' not in tmpStr and 'victim' not in tmpStr \
                    and 'fatal' not in tmpStr and 'wound' not in tmpStr:
                        tokens[j] = token.replace('woundedNum', 'TAG')
                        cnt += 1
                    total += 1

            outFile.write(f[i-1]+'\n')
            outFile.write(' '.join(tokens)+'\n')


print "Replaced ", cnt, "occurences out of ", total 