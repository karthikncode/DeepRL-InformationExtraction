''' script to find all shooter names'''

import sys, pdb

CONTEXT = 5

cnt = 0
total = 0

shooterNames = set()

with open(sys.argv[1], 'r') as inFile:
    with open(sys.argv[2], 'w') as outFile:
        f = inFile.read().split('\n')
        for i in range(1, len(f), 2):
            tokens = f[i].split()

            for j, token in enumerate(tokens):
                if 'shooterName' in token:
                    shooterNames.add(token.split('_')[0].lower())


        #write the shooter names to file
        outFile.write(str(list(shooterNames)).replace("'", '"'))



print "Got ", len(shooterNames), "names."