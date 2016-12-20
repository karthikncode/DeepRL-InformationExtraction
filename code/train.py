import numpy as np
from sklearn.linear_model import LogisticRegression
import scipy.sparse
import time
import itertools
import sys
import pickle
import helper
import constants

int2tags = constants.int2tags
int2tagsTag = ["TAG"] + int2tags
tags2int = constants.tags2int
tags = range(len(tags2int))

# main loop
def main(training_file,trained_model,previous_n,next_n, c, prune, test_file):
    helper.load_constants()
    train_data, identifier = load_data(training_file)
 
    test_data, test_ident = load_data(test_file)

    ## extract features
    tic = time.clock()
    print "get word_vocab"
    num_words, word_vocab = get_word_vocab(train_data, prune)
    print "feature extract for train"
    trainX, trainY = get_feature_matrix_n(previous_n,next_n,train_data, num_words, word_vocab, helper.other_features)
    print 'feature extract for test'
    testX, testY   = get_feature_matrix_n(previous_n, next_n, test_data, num_words, word_vocab, helper.other_features)    
    print time.clock()-tic

    ## train LR
    print("training")
    tic = time.clock()
    clf = LogisticRegression(C=c, multi_class='multinomial', solver='lbfgs')
    clf.fit(trainX,trainY)
    print time.clock()-tic

    print "predicting"
    predictY = clf.predict(testX)
    assert len(predictY) == len(testY)

    print "evaluating"

    evaluatePredictions(predictY, testY)

    feature_list = (word_vocab.keys() + helper.other_features) * (previous_n+next_n+1)  + word_vocab.keys() + ['previous_one'] * len(tags) + ['previous_two'] * len(tags)+ ['previous_three'] * len(tags)
    # getTopFeatures(clf,tags,feature_list)
    if trained_model != "":
        pickle.dump([clf, previous_n,next_n, word_vocab,helper.other_features], open( trained_model, "wb" ) )
    return [clf, previous_n,next_n, word_vocab,helper.other_features]

def evaluatePredictions(predicted, gold):
    range_toks = range(len(predicted))
    num_tags = len(int2tagsTag)
    correct = [0] * num_tags
    guessed = [0] * num_tags
    gold_c  = [0] * num_tags
    for i in range(len(int2tagsTag)):
        correct[i] = sum([predicted[j] == gold[j] and gold[j] == i for j in range_toks])
        guessed[i] = sum([predicted[j] == i for j in range_toks])
        gold_c[i] = sum([gold[j] == i for j in range_toks])

    helper.printScores(correct, guessed, gold_c)

def get_word_vocab(data, prune):
    num_words = 0
    word_vocab = {}
    for sentence in data:
        words_in_sentence = set()
        for word in sentence[0]:
            if word.lower() in words_in_sentence:
                continue
            if word.lower() not in word_vocab:
                word_vocab[word.lower()] = 1
            else:
                word_vocab[word.lower()] += 1
            words_in_sentence.add(word.lower())
        num_words += len(sentence[0])
    feature_list = []
    prune_features(word_vocab,feature_list, prune)
    return num_words,word_vocab

# reduce dimensions by removing features that don't appear often (not used)
def prune_features(feature_vocab, featureList, prune):
    for w in feature_vocab.keys():
        if feature_vocab[w] <= prune:
            feature_vocab.pop(w,None)
    index = 0
    for w in feature_vocab.keys():
        feature_vocab[w] = index
        featureList.append(w)
        index += 1

# get feature matrix given a list of sentences and tags (used for training) n: previous n words
def get_feature_matrix_n(previous_n,next_n,data, num_words, word_vocab, other_features,first_n=10):
    num_features = len(word_vocab) + len(other_features)
    total_features = (previous_n+next_n+1)*num_features + len(word_vocab) + previous_n * len(tags) + first_n
    #print num_words, num_features, total_features
    dataY = np.zeros(num_words)
    dataX = scipy.sparse.lil_matrix((num_words, total_features))
    curr_word = 0
    for sentence in data:
        other_words_lower = set([s.lower for s in sentence[0]])
        for i in range(len(sentence[0])):
            word = sentence[0][i]
            word_lower = word.lower()
            if word_lower in word_vocab:
                dataX[curr_word,word_vocab[word_lower]] = 1
                for j in range(previous_n):
                    if i+j+1<len(sentence[0]):
                        dataX[curr_word+j+1,(j+1)*num_features+word_vocab[word_lower]] = 1
                for j in range(next_n): 
                    if i-j-1 >= 0:
                        dataX[curr_word-j-1,(previous_n+j+1)*num_features+word_vocab[word_lower]] = 1
            for (index, feature_func) in enumerate(other_features):
                if feature_func(word):
                    dataX[curr_word,len(word_vocab)+index] = 1
                    for j in range(previous_n):
                        if i + j + 1 < len(sentence[0]):
                            dataX[curr_word+j+1,(j+1)*num_features+len(word_vocab)+index] = 1
                    for j in range(next_n):
                        if i - j - 1 >= 0:
                            dataX[curr_word-j-1,(previous_n+j+1)*num_features+len(word_vocab)+index] = 1
            for other_word_lower in other_words_lower:
                if other_word_lower != word_lower and other_word_lower in word_vocab:
                    dataX[curr_word,(previous_n+next_n+1)*num_features + word_vocab[other_word_lower]] = 1
            for j in range(previous_n):
                if j < i:
                    dataX[curr_word,(previous_n+next_n+1)*num_features+len(word_vocab)+len(tags) * j + dataY[curr_word-j-1]] = 1
            if i < first_n:
                dataX[curr_word,(previous_n+next_n+1)*num_features + len(word_vocab) + previous_n * len(tags)+i] = 1
            assert len(sentence[0]) == len(sentence[1])
            dataY[curr_word] = sentence[1][i]
            curr_word += 1
    return dataX, dataY

# split sentence into a list of words and a list of tags
def separate_word_tag(sentence):
    parts = sentence.split()
    words = []
    tags = []
    i = 0
    for index, part in enumerate(parts):
        part = part.strip()
        if len(part.split("_")) < 2 or part[0] == "_":
            continue
        i+=1
        tags.append(tags2int[part.split("_")[-1]])
        words.append(part.split("_")[0])
      
    return [words,tags]

# return a list of raw sentences (unprocessed)
def load_data(filename):
    sentence_list = [line.rstrip('\n') for line in open(filename)][1::2]
    identifier = [line.rstrip('\n') for line in open(filename)][::2]
    return map(separate_word_tag,sentence_list), identifier

# prints a list of top 10 features for each class
def getTopFeatures(clf, tags, featureList):
    A = np.copy(clf.coef_)
    for i in tags:
        print int2tags[i]
        #A[i] = map(abs, A[i])
        indices = np.argsort(A[i])[-20:][::-1]
        print indices
        for j in indices:
            print featureList[j]

def save_list_first_names(infile_path,outfile_path):
    l = set()
    with open(infile_path) as infile:
        for line in infile:
            l.add(line.split()[0].lower())
    print len(l)
    pickle.dump(l,open(outfile_path, "wb" ))


if __name__ == "__main__":

    mode = constants.mode
    if mode == 'EMA':
        training_file = "../data/tagged_data/EMA/train.tag"
        test_file = "../data/tagged_data/EMA/dev.tag"
    elif mode == "Shooter":
        training_file = "../data/tagged_data/Shootings/train.tag"
        test_file = "../data/tagged_data/Shootings/dev.tag"

    evaluate = True # Set true to score classifier on dev
    if not evaluate:
        test_file = False

    trained_model = "trained_model.large.y.p" #sys.argv[2]
    if constants.mode == "EMA":
        previous_n = 2 #sys.argv[3]
        next_n = 2
    else:
        previous_n = 0 #sys.argv[3]
        next_n = 4
    c = 10
    prune = 5
    main(training_file,trained_model,previous_n,next_n,c,prune, test_file)
