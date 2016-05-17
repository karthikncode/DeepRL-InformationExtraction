import train as train
import time, random
import scipy.sparse
import pycrfsuite as crf
import helper
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.tag import pos_tag
import random
import text2num as t2n

def trainModel(holdback=-1):
    ## extract features
    trainer = crf.Trainer(verbose=True)

    for xseq, yseq in zip(trainX, trainY):
        trainer.append(xseq, yseq, group = 0)

    for xseq, yseq in zip(testX, testY):
        trainer.append(xseq, yseq, group = 1)

    trainer.set_params({
        'c1': 2.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        # include transitions that are possible, but not observed
        'max_iterations': 250,  # stop earlier
        'feature.possible_transitions': True,
        'feature.possible_states': True,
    })
    trainer.train(trained_model, holdback)
    return trainer


def predict():
    tagger = crf.Tagger()
    tagger.open(trained_model)
    
    predictedY  =  []
    confidences =  []
    confidences_beam = []
    
    for xseq in testX:  
        yseq = tagger.tag(xseq)
        predictedY.append(yseq)
        confidences.append([tagger.marginal(yseq[i],i) for i in range(len(yseq))])   
        confidences_beam.append([ [tagger.marginal(tag, i)  for tag in train.int2tags]   for i in range(len(yseq))])
    return predictedY, testY, confidences, confidences_beam, tagger.info()


def predict(article, trained_model):
    tagger = crf.Tagger()
    tagger.open(trained_model)

    xseq = articleFeatureExtract(article)
    yseq  =  tagger.tag(xseq)

    confidences =  [tagger.marginal(yseq[i],i) for i in range(len(yseq))]
    confidences_beam = [ [tagger.marginal(tag, i)  for tag in train.int2tags]   for i in range(len(yseq))]
     
    return yseq, confidences



def featureExtract(data, identifier,  prev_n = 4, next_n = 4):
    features = []
    labels   = []
    int2tags = ["TAG"] + train.int2tags
    for index in range(len(data)):
        article = data[index][0]
        article_labels = [int2tags[t] for t in data[index][1]]
        article_features = articleFeatureExtract(article, prev_n, next_n)
        features.append(article_features)
        labels.append(article_labels)
    return features, labels

def articleFeatureExtract(article, prev_n = 4, next_n = 4):
    article_features = []
    title_features = {}
    labels = []
    # if '.' in article:
        # title = article[:article.index('.')]
        # for i in range(len(title)):
        #     t = title[i]
        #     tf = {}
        #     tf[t] = 1
        #     title_features[t] = 1
    for token_ind in range(len(article)):
        token = article[token_ind]
        context = {}
        for i in range(max(0, token_ind - prev_n), min(token_ind + next_n, len(article))):
            context_token = article[i]
            context[context_token] =1
        token_features = {}
        token_features["context"] = context
        # token_features["title"] = title_features
        token_features["token"] = token
        token_features[token]   = 1
        token_features["other"] = helper.getOtherFeatures(token)
        article_features.append(token_features)

    return article_features


if __name__ == '__main__':
    ##SCRIPT
    print "reload helper"
    reload(helper)
    helper.load_constants()
    print "end load helper"

    retrain =  True
    if retrain:
        num_blocks = 1
        ## num_blocks = 5
        training_file = "../data/tagged_data/EMA/train.tag"
        dev_file      = "../data/tagged_data/EMA/dev.tag"
        test_file      = "../data/tagged_data/EMA/test.tag"

        trained_model = "trained_model_crf.EMA.p"
        print "load files"
        train_data, train_identifier = train.load_data(training_file)
        test_data, test_identifier = train.load_data(dev_file)
        print "End load files"
        prev_n = 2
        next_n = 2
        print "Start Feature extract on train set"
        trainX, trainY = featureExtract(train_data,train_identifier, prev_n, next_n )
        print "Done Feature extract on train set"
        #trainX, trainY = featureExtract(dev_data, prev_n, next_n)
        print "Start Feature extract on test set"
        testX, testY = featureExtract(test_data, test_identifier, prev_n, next_n)
        print "Done Feature extract on test set"
        #testX, testY = featureExtract(train_data[split_index:], prev_n, next_n)
        trainer = trainModel(1)

