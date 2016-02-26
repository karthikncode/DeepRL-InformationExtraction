import train as train
import time
import scipy.sparse
import pycrfsuite as crf
import helper
from nltk.tokenize import sent_tokenize, word_tokenize







def trainModel():
    ## extract features
    trainer = crf.Trainer(verbose=True)
    
    for xseq, yseq in zip(trainX, trainY):
        trainer.append(xseq, yseq, group = 0)

    for xseq, yseq in zip(testX, testY):
        trainer.append(xseq, yseq, group = 1)
        
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 0,  # coefficient for L2 penalty
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True,
        'feature.possible_states': True,
    })
    trainer.train(trained_model)
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



def featureExtract(data):
    features = []
    labels   = []
    for article, article_labels in data:
        article_features = []
        if '.' in article:
            title = article[:article.index('.')]
        title_features = {}
        for t in title:
#             t = t.lower()
            title_features[t] = 1
        for token_ind in range(len(article)):
            token = article[token_ind]
            context = {}
            prev_n = 4
            next_n = 4
            for i in range(max(0, token_ind - prev_n), min(token_ind + next_n, len(article))):
                context_token = article[i]
                context[context_token] = 1
                context["other"] = helper.getOtherFeatures(context_token)
                context["token"] = context_token
#             token = token.lower()
            token_features = {}
            token_features["context"] = context
            token_features["title"] = title_features
            token_features["token"] = token
            token_features[token]   = 1
            other_features = helper.getOtherFeatures(token)
            token_features["other"] = helper.getOtherFeatures(token)
            article_features.append(token_features)
        features.append(article_features)
        labels.append([train.int2tags[tag] for tag in article_labels])

    return features, labels

def articleFeatureExtract(article):
    article_features = []
    for token_ind in range(len(article)):
        token = article[token_ind]
        context = {}
        prev_n = 4
        next_n = 4
        for i in range(max(0, token_ind - prev_n), min(token_ind + next_n, len(article))):
            context_token = article[i]
            context[context_token] = 1
            context["other"] = helper.getOtherFeatures(context_token)
            context["token"] = context_token
            token_features = {}
            token_features["context"] = context
            token_features["title"] = title_features
            token_features["token"] = token
            token_features[token]   = 1
            other_features = helper.getOtherFeatures(token)
            token_features["other"] = helper.getOtherFeatures(token)
            article_features.append(token_features)
        features.append(article_features)
    return article_features


training_file = "../data/tagged_data/whole_text_full_city2/train.tag"
trained_model = "trained_model_crf.p"


helper.load_constants()
#Load data and split into train/dev
all_data, all_identifier = train.load_data(training_file)
train_split = .6
split_index = int(len(all_data)*train_split)

train_data, train_identifier = all_data[:split_index], all_identifier[:split_index]
balanced_train_data  = balance_data() #Balance data to handle skew

test_data, test_identifier = all_data[split_index:], all_identifier[split_index:]


#Feature extraction
trainX, trainY = featureExtract(balanced_train_data)
testX, testY = featureExtract(test_data )

retrain = True
if retrain:
    trainer = trainModel()
