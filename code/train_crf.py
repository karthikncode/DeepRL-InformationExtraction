import train as train
import time, random
import scipy.sparse
import pycrfsuite as crf
import helper
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.tag import pos_tag

def trainModel(holdback=-1):
    ## extract features
    trainer = crf.Trainer(verbose=True)

    for xseq, yseq in zip(trainX, trainY):
        trainer.append(xseq, yseq, group = 0)

    for xseq, yseq in zip(testX, testY):
        trainer.append(xseq, yseq, group = 1)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        # include transitions that are possible, but not observed
        'max_iterations': 200,  # stop earlier
        #'feature.possible_transitions': True,
        #'feature.possible_states': True,
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

def balance_data():
    filtered_data = []
    for text, tags in train_data:
        article = " ".join(text)
        filtered_article = []
        filtered_tags = []
        sentences = sent_tokenize(article)
        start_index = 0
        end_index = 0
        for sentence in sentences:
            start_index = end_index
            words = sentence.split(" ")
            for word in words:
                if not tags[end_index] == 0:
                    end_index = start_index + len(words)
                    filtered_article += text[start_index: end_index]
                    filtered_tags += tags[start_index: end_index]
                    assert len(tags[start_index: end_index]) == len(text[start_index: end_index])
                    break
                end_index += 1

            assert len(filtered_tags) == len(filtered_article)
        filtered_data.append( [filtered_article, filtered_tags])
    return filtered_data


def featureExtract(data, prev_n = 4, next_n = 4):
    features = []
    labels   = []
    for article, article_labels in data:
        article_features = articleFeatureExtract(article, prev_n, next_n)
        features.append(article_features)
        labels.append([train.int2tags[tag] for tag in article_labels])

    return features, labels

def articleFeatureExtract(article, prev_n = 4, next_n = 4):
    article_features = []
    title_features = {}
#    pos_tags = pos_tag(article)
    if '.' in article:
        title = article[:article.index('.')]
        for t in title:
            other_title_features = helper.getOtherFeatures(t)
            other_title_features[t] = 1
            title_features[t] = other_title_features
    for token_ind in range(len(article)):
        token = article[token_ind]
        context = {}
        for i in range(max(0, token_ind - prev_n), min(token_ind + next_n, len(article))):
            context_token = article[i]
            context[context_token] = 1
            context["other"] = helper.getOtherFeatures(context_token)
            context["token"] = context_token
        token_features = {}
        #        token_features["pos_tag"] = pos_tags[token_ind][1]
        token_features["context"] = context
        token_features["title"] = title_features
        token_features["token"] = token
        token_features[token]   = 1
        other_features = helper.getOtherFeatures(token)
        token_features["other"] = helper.getOtherFeatures(token)
        if i < 10:
            token_features[str(i)] = True
        article_features.append(token_features)
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
    #test_data, test_identifier = train.load_data(test_file)
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
        prev_n = 3
        next_n = 3
        train_split = 1
        print "Calculate split index"
        split_index = int(train_split * len(train_data))
        print "Start Feature extract on train set"
        trainX, trainY = featureExtract(train_data[:split_index], prev_n, next_n )
        print "Done Feature extract on train set"
        #trainX, trainY = featureExtract(dev_data, prev_n, next_n)
        print "Start Feature extract on test set"
        testX, testY = featureExtract(dev_data, prev_n, next_n)
        print "Done Feature extract on test set"
        #testX, testY = featureExtract(train_data[split_index:], prev_n, next_n)
        trainer = trainModel(1)
        print "YALA with context size being", prev_n, next_n, "complet"

