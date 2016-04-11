''' script to consolidate vectorizers
    
    Input: files from dloads
    Output: into consolidated

'''
import numpy as np
import copy
import sys, json, pdb, pickle, operator, collections
from itertools import izip
import inflect
import predict as predict
import helper
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
ARTICLES2, TITLES2, IDENTIFIERS2, DOWNLOADED_ARTICLES2 = [],[],[],[] #just a placeholder for copy


fileName = sys.argv[1]
#IMP: lists must be of the form train.extra.0
numLists = int(sys.argv[2])
trained_model = sys.argv[3]
if not "crf" in trained_model:
    print "NOT CRF"
    trained_model = pickle.load( open(sys.argv[3], "rb" ) )
else:
    print "CRF"


#function to build the dictionary for words to be used for context features
def getContextDictionary(original_train_articles, context=3):
    articles = [' '.join(tokens) for tokens, tags in original_train_articles]
    
    vectorizer1 = CountVectorizer(min_df=1)
    vectorizer2 = TfidfVectorizer(min_df=1)
    vectorizer1.fit(articles)
    vectorizer2.fit(articles)

    print "Computed vectorizers."
    return vectorizer1, vectorizer2

def extractEntitiesWithConfidences(article):
    global trained_model
    #article is a list of words
    joined_article = ' '.join(article)
    pred, conf_scores, conf_cnts = predict.predictWithConfidences(trained_model, joined_article, False, helper.cities)

    for i in range(len(conf_scores)):
        if conf_cnts[i] > 0:
            conf_scores[i] /= conf_cnts[i]

    return pred.split(' ### '), conf_scores


########################### SCRIPT ########################################

if ',' in fileName:
    fileNames = fileName.split(',')
else:
    fileNames = [fileName]

globalIndx = 0

for fileName in fileNames:
    for listNum in range(numLists):
        print "LIST", listNum

        #read the file
        articles, titles, identifiers, downloaded_articles = loadFile(fileName+'.'+str(listNum))

        #need this information only once (original articles)
        if listNum==0:
            ARTICLES = articles
            TITLES = titles
            IDENTIFIERS = identifiers
            DOWNLOADED_ARTICLES = [[] for q in range(len(ARTICLES))]

            #calculate the context dictionary
            
            vectorizer1, vectorizer2 = getContextDictionary(articles)

#now to store everything
pickle.dump([vectorizer1, vectorizer2], open(sys.argv[4], "wb"))




