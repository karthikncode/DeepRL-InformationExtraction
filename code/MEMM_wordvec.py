import numpy as np
from sklearn.linear_model import LogisticRegression
import scipy.sparse
import time
import itertools
import sys
import re

tags2int = {"TAG": 0, "shooterName":1, "killedNum":2, "woundedNum":3, "city":4}
int2tags = ["TAG",'shooterName','killedNum','woundedNum','city']
tags = [0,1,2,3,4]


# main loop
def main(training_file,testing_file,outputfile,model_id,word2vec_file):
    train_data, identifier = load_data(training_file)
    test_data, identifier = load_data(testing_file)
    words, features = parse_word2vec(word2vec_file)
    n=0
    c=10
    viterbi = False
    if model_id == "2":
        c=10
        n = 1
    if model_id == '3':
        c=10
        n = 1
        viterbi = True

    ## extract features
    #tic = time.clock()
    trainX, trainY = get_feature_matrix_n(n,train_data,words,features)
    #print time.clock()-tic
    ## train LR
    print("training")
    tic = time.clock()
    clf = LogisticRegression(C=c)
    clf.fit(trainX,trainY)
    print time.clock()-tic

    print("predicting")
    tic = time.clock()
    f = open(outputfile,'w')
    for i in range(len(test_data)+len(identifier)):
        if i%2 == 1:
            y = predict_tags_n(viterbi, n, clf, test_data[i/2][0], words, features)
            f.write(" ".join([test_data[i/2][0][j]+"_"+int2tags[int(y[j])] for j in range(len(test_data[i/2][0]))]))
            f.write("\n")
        else:
            f.write(identifier[i/2])
            f.write("\n")
    f.close()
    print time.clock()-tic
    return clf

def get_feature_matrix_n(n,data,words,features):
    num_words = 0
    for sentence in data:
        num_words += len(sentence[0])
    num_features = len(features[0])
    total_features = (n+1)*num_features + n
    print num_words, num_features
    dataX = np.zeros((num_words,total_features))
    dataY = np.zeros(num_words)
    curr_word_index = 0
    for sentence in data:
        for i in range(len(sentence[0])):
            word = sentence[0][i].lower()
            if word in words:
                feature = features[words[word]]
            else:
                feature = np.zeros(num_features)
            for j in range(n+1):
                if curr_word_index + j < len(sentence):
                    dataX[curr_word_index+j,j*num_features:(j+1)*num_features] = feature
            dataY[curr_word_index] = sentence[1][i]
            curr_word_index+=1
    return dataX, dataY

def predict_tags_n(viterbi, n, clf, sentence, words, features):
    num_features = len(features[0])
    total_features = (n+1)*num_features + n
    dataX = np.zeros((len(sentence),total_features))
    dataY = np.zeros(len(sentence))

    for i in range(len(sentence)):
        word = sentence[i].lower()
        if word in words:
            feature = features[words[word]]
        else:
            feature = np.zeros(num_features)
        for j in range(n+1):
            if i + j < len(sentence):
                dataX[i+j,j*num_features:(j+1)*num_features] = feature
        for j in range(n):
            if j < i:
                dataX[i,(n+1)*num_features+j] = dataY[i-j-1]
            dataY[i] = clf.predict(dataX[i,:].reshape(1, -1))
    return dataY

# split sentence into a list of words and a list of tags
def separate_word_tag(sentence):
    parts = sentence.split()
    words = []
    tags = []
    i = 0
    for part in parts:
        i+=1
        if i > 10:
            break
        words.append(re.sub(r'__+','_',part).split("_")[0])
        if part.split("_")[1] == '':
            print part
        tags.append(tags2int[re.sub(r'__+','_',part).split("_")[1]])
    return [words,tags]

# return a list of raw sentences (unprocessed)
def load_data(filename):
    sentence_list = [line.rstrip('\n') for line in open(filename)][1::2]
    identifier = [line.rstrip('\n') for line in open(filename)][::2]
    return map(separate_word_tag,sentence_list), identifier

def parse_word2vec(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    features = np.zeros((len(lines), len(lines[0].split())-1))
    words = {}
    for i in range(len(lines)):
        line = lines[i].split()
        words[line[0]] = i
        features[i] = line[1:]
    return words,features
    

if __name__ == "__main__":
    training_file = "data/tagged_data/has_title/train.tag" #sys.argv[1]
    testing_file = "data/tagged_data/has_title/dev.tag" #sys.argv[2]
    model_id = 1#sys.argv[3]
    output_file = "output.tag"
    word2vec_file = "data/vectors_wiki.normalized.txt"
    main(training_file,testing_file,output_file,model_id, word2vec_file)
