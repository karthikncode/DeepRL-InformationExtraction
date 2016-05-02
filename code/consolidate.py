''' script to consolidate articles, entities, etc. into pickle files for fast loading
    
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
from server import loadFile, computeContext
import pdb

WORD_LIMIT = 1000
tfidf_vectorizer = TfidfVectorizer()

def dd():
    return {}

def ddd():
    return collections.defaultdict(dd)

#global vars
CONTEXT_LENGTH = 3
COSINE_SIM = collections.defaultdict(dd)
ENTITIES = collections.defaultdict(ddd)
CONFIDENCES = collections.defaultdict(ddd)

ARTICLES, TITLES, IDENTIFIERS, DOWNLOADED_ARTICLES = [],[],[],[]
ARTICLES2, TITLES2, IDENTIFIERS2, DOWNLOADED_ARTICLES2 = [],[],[],[] #just a placeholder for copy


fileName = sys.argv[1]
#IMP: lists must be of the form train.extra.0
numLists = int(sys.argv[2])
trained_model =  str(sys.argv[3])
if not "crf" in trained_model:
    trained_model = pickle.load( open(sys.argv[3], "rb" ) )

if len(sys.argv) > 5:
    print "Loading vectorizers from ", sys.argv[5]
    vectorizer1, vectorizer2 = pickle.load(open(sys.argv[5], "rb"))
else:
    print "Not loading any vectorizer"


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

    result = pred.split(' ### '), conf_scores
    split =  [" ### " in r for r in result]
    assert sum(split) == 0
    return result

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
        print "LEN ARTICLES", len(articles)
        print "final LEN IDENTIFIERS", len(IDENTIFIERS)

        #need this information only once (original articles)
        if listNum==0:
            ARTICLES = articles
            TITLES = titles
            IDENTIFIERS = identifiers
            DOWNLOADED_ARTICLES = [[] for q in range(len(ARTICLES))]

            #calculate the context dictionary
            if not vectorizer1:
                vectorizer1, vectorizer2 = getContextDictionary(articles)

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
            ENTITIES[indx+globalIndx][listNum][0], CONFIDENCES[indx+globalIndx][listNum][0] = entities, confidences

            #now for the downloaded extra articles
            for j, newArticle in enumerate(downloaded_articles[indx]):
                newArticle = newArticle.split(' ')[:WORD_LIMIT] 
                entities, confidences = extractEntitiesWithConfidences(newArticle)
                ENTITIES[indx+globalIndx][listNum][j+1], CONFIDENCES[indx+globalIndx][listNum][j+1] = entities, confidences
            # pdb.set_trace()
        print    

    globalIndx += len(articles)
    ARTICLES2 += ARTICLES
    TITLES2 += TITLES
    IDENTIFIERS2 += IDENTIFIERS
    DOWNLOADED_ARTICLES2 += DOWNLOADED_ARTICLES

ARTICLES, TITLES, IDENTIFIERS, DOWNLOADED_ARTICLES = ARTICLES2, TITLES2, IDENTIFIERS2, DOWNLOADED_ARTICLES2

# now to calculate cosine_sim using tf-idf calculated using all the downloaded articles
for indx, article in enumerate(ARTICLES):
    originalArticle = article[0]
    allArticles = [' '.join(originalArticle)] 
    for i in range(numLists):
        allArticles += DOWNLOADED_ARTICLES[indx][i]

    tfidf_matrix = tfidf_vectorizer.fit_transform(allArticles)
    cnt = 1
    for listNum in range(len(DOWNLOADED_ARTICLES[indx])):
        sublist = DOWNLOADED_ARTICLES[indx][listNum]
        if len(sublist)>0:
            COSINE_SIM[indx][listNum] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[cnt:cnt+len(sublist)])[0]
        else:
            print "not enough elements in sublist for cosine_sim"
            COSINE_SIM[indx][listNum] = []
        # pdb.set_trace()
        cnt += len(sublist)



#calculate the contexts
CONTEXT1 = collections.defaultdict(ddd)
CONTEXT2 = collections.defaultdict(ddd)
computeContext(ENTITIES, CONTEXT1, ARTICLES, DOWNLOADED_ARTICLES, vectorizer1, CONTEXT_LENGTH)
computeContext(ENTITIES, CONTEXT2, ARTICLES, DOWNLOADED_ARTICLES, vectorizer2, CONTEXT_LENGTH)
print "final LEN ARTICLES", len(ARTICLES)
print "final LEN IDENTIFIERS", len(IDENTIFIERS)

#now to store everything
pickle.dump([ARTICLES, TITLES, IDENTIFIERS, DOWNLOADED_ARTICLES, ENTITIES, CONFIDENCES, COSINE_SIM, CONTEXT1, CONTEXT2], open(sys.argv[4], "wb"))




