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
    print article
    xseq = articleFeatureExtract(article)
    print xseq
    yseq  =  tagger.tag(xseq)
    confidences =  [tagger.marginal(yseq[i],i) for i in range(len(yseq))]
    confidences_beam = [ [tagger.marginal(tag, i)  for tag in train.int2tags]   for i in range(len(yseq))]
     
    return yseq, confidences



def featureExtract(data, identifier,  prev_n = 4, next_n = 4):
    features = []
    labels   = []
    for index in range(len(data)):
        article = data[index][0]
        gold    = identifier[index].split(',')[:4]
        article_features, article_labels = articleFeatureExtract(article, gold, prev_n, next_n)
        features.append(article_features)
        labels.append(article_labels)
    return features, labels

def articleFeatureExtract(article, gold=None, prev_n = 4, next_n = 4):
    article_features = []
    title_features = {}
    labels = []
#    pos_tags = pos_tag(article)
    if '.' in article:
        title = article[:article.index('.')]
        for i in range(len(title)):
            t = title[i]
            tf = {}
            tf[t] = 1
        #    tf["other"] = helper.getOtherFeatures(t)
            title_features[t] = 1
    for token_ind in range(len(article)):
        token = article[token_ind]
        context = {}
        for i in range(max(0, token_ind - prev_n), min(token_ind + next_n, len(article))):
            context_token = article[i]
            #if i == token_ind:
            #continue
            #c = {}
            context[context_token] =1
            #c["other"] = helper.getOtherFeatures(token)
            #if i < 10:
            #    c[str(token_ind)] = 1
            #context[ str(i - token_ind)] = c
        token_features = {}
        token_features["context"] = context
        token_features["title"] = title_features
        token_features["token"] = token
        token_features[token]   = 1
        token_features["other"] = helper.getOtherFeatures(token)
        #if token_ind < 10:
        #    token_features[str(token_ind)] = 1
        article_features.append(token_features)

        if gold:
            clean_token = token.strip().lower()
            if helper.is_number_word(clean_token):
                clean_token = str(t2n.text2num(clean_token))

            label_l = []
            label = ''
            for i in range(len(gold)):
                gold_ent = gold[i].strip().lower()
                if gold_ent == '' or gold_ent == '0':
                    continue
                tag = train.int2tags[i+1]
                if i >0:
                    if gold_ent == clean_token:
                        label_l.append(tag)

                    if i == 3 and len(gold_ent.split()) > 1:
                        for g in gold_ent.split():
                            if g == clean_token:
                                label_l.append(tag)
                                break
                else:
                    gold_set = set(gold_ent.split('|'))
                    if clean_token in gold_set:
                        label_l.append(tag)
            if len(label_l) > 1:
                print label_l
                print "gold", gold, "token", clean_token
                label = label_l[0]
            elif len(label_l) == 1:
                label = label_l[0]
            else:
                label = "TAG"
            labels.append(label)
    if gold:
        return article_features, labels
    else:
        return article_features

##SCRIPT
print "reload helper"
reload(helper)
helper.load_constants()
print "end load helper"

retrain =  True
if retrain:
    num_blocks = 1
    ## num_blocks = 5
    training_file = "../data/tagged_data/whole_text_full_city2/train.tag"
    dev_file      = "../data/tagged_data/whole_text_full_city2/dev.tag"
    test_file      = "../data/tagged_data/whole_text_full_city2/test.tag"

    trained_model = "trained_model_crf.p"
    print "load files"
    train_data, train_identifier = train.load_data(training_file)
    dev_data, dev_identifier = train.load_data(dev_file)
    print "End load files"
    test_data, test_identifier = train.load_data(test_file)
    #all_data = train_data + dev_data + test_data
    #all_identifier = train_identifier + dev_identifier + test_identifier
    #len_all_data = len(all_data)
    #indices = range(len_all_data)
    for j in range(num_blocks):
        # start_test_index = (len_all_data / num_blocks) * j
        # end_test_index   = (len_all_data / num_blocks) * (j+1)
        # train_indices = indices[:start_test_index]
        # if j < 4:
        #     train_indices += indices[end_test_index:]
        # test_indices = indices[start_test_index:end_test_index]

        # train_data_block, train_identifier_block =[all_data[i] for i in train_indices], [all_identifier[i] for i in train_indices]
        
        # test_data_block, test_identifier_block =  [all_data[i] for i in test_indices], [all_identifier[i] for i in test_indices]
        # test_data, test_identifier = train.load_data(test_file)

        #Feature extraction
        #trainX, trainY = featureExtract(train_data_block)
        #testX, testY = featureExtract(test_data_block )
        prev_n = 2
        next_n = 2
        train_split = 1
        print "Start Feature extract on train set"
        trainX, trainY = featureExtract(train_data,train_identifier, prev_n, next_n )
        print "Done Feature extract on train set"
        #trainX, trainY = featureExtract(dev_data, prev_n, next_n)
        print "Start Feature extract on test set"
        testX, testY = featureExtract(dev_data, dev_identifier, prev_n, next_n)
        print "Done Feature extract on test set"
        #testX, testY = featureExtract(train_data[split_index:], prev_n, next_n)
        trainer = trainModel(1)
        print "YALA with context size being", prev_n, next_n, "complet"

