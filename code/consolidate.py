''' script to consolidate articles, entities, etc. '''
import numpy as np
import copy
import sys, json, pdb, pickle, operator, collections
from itertools import izip
import inflect
import predict2 as predict
import helper
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from random import shuffle
from operator import itemgetter
from server import loadFile

WORD_LIMIT = 1000
tfidf_vectorizer = TfidfVectorizer()

def dd():
    return {}

def ddd():
    return collections.defaultdict(dd)

#global vars
COSINE_SIM = collections.defaultdict(dd)
ENTITIES = collections.defaultdict(ddd)
CONFIDENCES = collections.defaultdict(ddd)

ARTICLES, TITLES, IDENTIFIERS, DOWNLOADED_ARTICLES = [],[],[],[]

fileName = sys.argv[1]
#IMP: lists must be of the form train.extra.0
numLists = int(sys.argv[2])
trained_model = pickle.load( open(sys.argv[3], "rb" ) )



def extractEntitiesWithConfidences(article):
    global trained_model
    #article is a list of words
    joined_article = ' '.join(article)
    pred, conf_scores, conf_cnts = predict.predictWithConfidences(trained_model, joined_article, False, helper.cities)

    for i in range(len(conf_scores)):
        if conf_cnts[i] > 0:
            conf_scores[i] /= conf_cnts[i]

    return pred.split(','), conf_scores




for listNum in range(numLists):
    print "LIST", listNum

    #read the file
    articles, titles, identifiers, downloaded_articles = loadFile(fileName+'.'+str(listNum))

    #need this information only once
    if listNum==0:
        ARTICLES = articles
        TITLES = titles
        IDENTIFIERS = identifiers
        DOWNLOADED_ARTICLES = [[] for q in range(len(ARTICLES))]

    for indx in range(len(articles)):
        DOWNLOADED_ARTICLES[indx].append(downloaded_articles[indx])

    assert(len(articles)>0 and len(ARTICLES) == len(articles))

    print "Calculating ENTITIES and CONFIDENCES...\n"
    #extract entities and save them
    for indx, article in enumerate(articles):
        print indx,'/',len(articles)
        #IMP: adding original article to all ENTITIES and CONFIDENCES LISTS at 0th position
        originalArticle = article[0]
        entities, confidences = extractEntitiesWithConfidences(originalArticle)
        ENTITIES[indx][listNum][0], CONFIDENCES[indx][listNum][0] = entities, confidences

        #now for the downloaded extra articles
        for j, newArticle in enumerate(downloaded_articles[indx]):
            newArticle = newArticle.split(' ')[:WORD_LIMIT] 
            entities, confidences = extractEntitiesWithConfidences(newArticle)
            ENTITIES[indx][listNum][j+1], CONFIDENCES[indx][listNum][j+1] = entities, confidences
        # pdb.set_trace()
    print    

# now to calculate cosine_sim using tf-idf calculated using all the downloaded articles
for indx, article in enumerate(ARTICLES):
    originalArticle = article[0]
    allArticles = [' '.join(originalArticle)] 
    for i in range(numLists):
        allArticles += DOWNLOADED_ARTICLES[indx][i]

    tfidf_matrix = tfidf_vectorizer.fit_transform(allArticles)

    cnt = 1
    for listNum, sublist in enumerate(DOWNLOADED_ARTICLES[indx]):
        if len(sublist)>0:
            COSINE_SIM[indx][listNum] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[cnt:cnt+len(sublist)])[0]
        else:
            print "not enough elements in sublist for cosine_sim"
            COSINE_SIM[indx][listNum] = []
        # pdb.set_trace()
        cnt += len(sublist)


#now to store everything
pickle.dump([ARTICLES, TITLES, IDENTIFIERS, DOWNLOADED_ARTICLES, ENTITIES, CONFIDENCES, COSINE_SIM], open(sys.argv[4], "wb"))




