import zmq, time
import numpy as np
import copy
import sys, json, pdb, pickle, operator, collections
import helper
import predict as predict
from train import load_data
from itertools import izip
import inflect
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from random import shuffle
from operator import itemgetter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from classifier import Classifier
import constants
import re

DEBUG = False
ANALYSIS = False
COUNT_ZERO = False

#Global variables
int2tags = constants.int2tags

NUM_ENTITIES = len(int2tags)
NUM_QUERY_TYPES = NUM_ENTITIES + 1
WORD_LIMIT = 1000
CONTEXT_LENGTH = 3
CONTEXT_TYPE = None
STATE_SIZE = 4*NUM_ENTITIES+1 + 2*CONTEXT_LENGTH*NUM_ENTITIES
STOP_ACTION = NUM_ENTITIES
IGNORE_ALL = STOP_ACTION + 1
ACCEPT_ALL = 999 #arbitrary


trained_model = None
tfidf_vectorizer = TfidfVectorizer()
inflect_engine = inflect.engine()

def dd():
    return {}

def ddd():
    return collections.defaultdict(dd)

#global caching to speed up
# TRAIN_TFIDF_MATRICES = {}
# TRAIN_ENTITIES = collections.defaultdict(dd)
# TRAIN_CONFIDENCES = collections.defaultdict(dd)

# TEST_TFIDF_MATRICES = {}
# TEST_ENTITIES = collections.defaultdict(dd)
# TEST_CONFIDENCES = collections.defaultdict(dd)

TRAIN_COSINE_SIM = collections.defaultdict(dd)
TRAIN_ENTITIES = collections.defaultdict(ddd)
TRAIN_CONFIDENCES = collections.defaultdict(ddd)
TRAIN_CONTEXT = collections.defaultdict(ddd) #final value will be a vector


TEST_COSINE_SIM =  collections.defaultdict(dd)
TEST_ENTITIES = collections.defaultdict(ddd)
TEST_CONFIDENCES = collections.defaultdict(ddd)
TEST_CONTEXT = collections.defaultdict(ddd) #final value will be a vector

CORRECT = collections.defaultdict(lambda:0.)
GOLD = collections.defaultdict(lambda:0.)
PRED = collections.defaultdict(lambda:0.)
EVALCONF = collections.defaultdict(lambda:[])
EVALCONF2 = collections.defaultdict(lambda:[])
QUERY = collections.defaultdict(lambda:0.)
ACTION = collections.defaultdict(lambda:0.)
CHANGES = 0
evalMode = False
STAT_POSITIVE, STAT_NEGATIVE = 0, 0 #stat. sign.

CONTEXT = None

def splitBars(w):
    return [q.strip() for q in w.split('|')]

#Environment for each episode
class Environment:
    def __init__(self, originalArticle, newArticles, goldEntities, indx, args, evalMode):
        self.indx = indx
        self.originalArticle = originalArticle
        self.newArticles = newArticles #extra articles to process
        self.goldEntities = goldEntities
        self.ignoreDuplicates = args.ignoreDuplicates
        self.entity = args.entity
        self.aggregate = args.aggregate
        self.delayedReward = args.delayedReward
        self.shooterLenientEval = args.shooterLenientEval
        self.listNum = 0 #start off with first list    
        self.rlbasicEval = args.rlbasicEval    
        self.rlqueryEval = args.rlqueryEval    

        self.shuffledIndxs = [range(len(q)) for q in self.newArticles]
        if not evalMode and args.shuffleArticles:
            for q in self.shuffledIndxs:
                shuffle(q)

        self.state = [0 for i in range(STATE_SIZE)]
        self.terminal = False

        self.bestEntities = collections.defaultdict(lambda:'') #current best entities
        self.bestConfidences = collections.defaultdict(lambda:0.)
        self.bestEntitySet = None
        if self.aggregate == 'majority':
            self.bestEntitySet = collections.defaultdict(lambda:[])
        self.bestIndex = (0,0)
        self.prevListNum = 0
        self.prevArticleIndx = 0

        # to keep track of extracted values from previousArticle
        # start off with list 0 always
        if 0 in ENTITIES[self.indx][0]:
            self.prevEntities, self.prevConfidences = ENTITIES[self.indx][0][0], CONFIDENCES[self.indx][0][0]
        else:
            self.prevEntities, self.prevConfidences = self.extractEntitiesWithConfidences(self.originalArticle)
            ENTITIES[self.indx][0][0] = self.prevEntities
            CONFIDENCES[self.indx][0][0] = self.prevConfidences

        #store the original entities before updating state
        self.originalEntities = self.prevEntities


        #calculate tf-idf similarities using all the articles related to the original
        self.allArticles = [originalArticle] + [item for sublist in self.newArticles for item in sublist]
        self.allArticles = [' '.join(q) for q in self.allArticles]

        if self.indx not in COSINE_SIM:
            # self.tfidf_matrix = TFIDF_MATRICES[0][self.indx]
            self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.allArticles)
            cnt = 0
            for listNum, sublist in enumerate(self.newArticles):
                COSINE_SIM[self.indx][listNum] = cosine_similarity(self.tfidf_matrix[0:1], self.tfidf_matrix[cnt:cnt+len(sublist)])[0]
                pdb.set_trace()
                cnt += len(sublist)

        #update the initial state
        self.stepNum = [0 for q in range(NUM_QUERY_TYPES)]
        
        self.updateState(ACCEPT_ALL, 1, self.ignoreDuplicates) 

        return

    def extractEntitiesWithConfidences(self, article):
        #article is a list of words
        joined_article = ' '.join(article)
        pred, conf_scores, conf_cnts = predict.predictWithConfidences(trained_model, joined_article, False, helper.cities)

        for i in range(len(conf_scores)):
            if conf_cnts[i] > 0:
                conf_scores[i] /= conf_cnts[i]

        return pred.split(' ### '), conf_scores

    #find the article similarity between original and newArticle[i] (=allArticles[i+1])
    def articleSim(self, indx, listNum, i):
        # return cosine_similarity(self.tfidf_matrix[0:1], self.tfidf_matrix[i+1:i+2])[0][0]
        return COSINE_SIM[indx][listNum][i]

    # update the state based on the decision from DQN
    def updateState(self, action, query, ignoreDuplicates=False):
        global CONTEXT, CONTEXT_TYPE

        #use query to get next article
        articleIndx = None

        if self.rlbasicEval:
            #ignore the query decision from the agent
            listNum = self.listNum
            self.listNum += 1
            if self.listNum == NUM_QUERY_TYPES: self.listNum = 0
        else:
            listNum = query-1 #convert from 1-based to 0-based         

        if self.rlqueryEval:
            #set the reconciliation action
            action = ACCEPT_ALL        

        if ignoreDuplicates:
            nextArticle = None
            while not nextArticle and self.stepNum[listNum] < len(self.newArticles[listNum]):
                articleIndx = self.shuffledIndxs[listNum][self.stepNum]
                if self.articleSim(self.indx, listNum, articleIndx) < 0.95:
                    nextArticle = self.newArticles[listNum][articleIndx]
                else:
                    self.stepNum[listNum] += 1                
        else:
            #get next article            
            if self.stepNum[listNum] < len(self.newArticles[listNum]):
                articleIndx = self.shuffledIndxs[listNum][self.stepNum[listNum]]
                nextArticle = self.newArticles[listNum][articleIndx]
            else:
                nextArticle = None

        if action != STOP_ACTION:
            # integrate the values into the current DB state
            entities, confidences = self.prevEntities, self.prevConfidences

            # all other tags
            for i in range(NUM_ENTITIES):
                if action != ACCEPT_ALL and i != action: continue #only perform update for the entity chosen by agent

                self.bestIndex = (self.prevListNum, self.prevArticleIndx) #analysis
                if self.aggregate == 'majority':
                    self.bestEntitySet[i].append((entities[i], confidences[i]))
                    self.bestEntities[i], self.bestConfidences[i] = self.majorityVote(self.bestEntitySet[i])
                else:
                    if i==0:
                        #handle shooterName -  add to list or directly replace
                        if not self.bestEntities[i]:
                            self.bestEntities[i] = entities[i]
                            self.bestConfidences[i] = confidences[i]
                        elif self.aggregate == 'always' or confidences[i] > self.bestConfidences[i]:
                            self.bestEntities[i] = entities[i] #directly replace
                            # self.bestEntities[i] = self.bestEntities[i] + '|' + entities[i] #add to list
                            self.bestConfidences[i] = confidences[i]
                    else:
                        if not self.bestEntities[i] or self.aggregate == 'always' or confidences[i] > self.bestConfidences[i]:
                            self.bestEntities[i] = entities[i]
                            self.bestConfidences[i] = confidences[i]
                            # print "Changing best Entities"
                            # print "New entities", self.bestEntities
            if DEBUG:
                print "entitySet:", self.bestEntitySet

        if nextArticle and action != STOP_ACTION:
            assert(articleIndx != None)
            if (articleIndx+1) in ENTITIES[self.indx][listNum]:
                entities, confidences = ENTITIES[self.indx][listNum][articleIndx+1], CONFIDENCES[self.indx][listNum][articleIndx+1]
            else:
                entities, confidences = self.extractEntitiesWithConfidences(nextArticle)
                ENTITIES[self.indx][listNum][articleIndx+1], CONFIDENCES[self.indx][listNum][articleIndx+1] = entities, confidences
            assert(len(entities) == len(confidences))
        else:
            # print "No next article"
            entities, confidences = [""]*NUM_ENTITIES, [0]*NUM_ENTITIES
            self.terminal = True

        #modify self.state appropriately
        # print(self.bestEntities, entities)
        if constants.mode == 'Shooter':
            matches = map(self.checkEquality, self.bestEntities.values()[1:-1], entities[1:-1])
            matches.insert(0, self.checkEqualityShooter(self.bestEntities.values()[0], entities[0]))
            matches.append(self.checkEqualityCity(self.bestEntities.values()[-1], entities[-1]))
        else:
            matches = map(self.checkEqualityShooter, self.bestEntities.values(), entities)

        # pdb.set_trace()
        self.state = [0 for i in range(STATE_SIZE)]
        for i in range(NUM_ENTITIES):
            self.state[i] = self.bestConfidences[i] #DB state
            self.state[NUM_ENTITIES+i] = confidences[i]  #IMP: (original) next article state
            matchScore = float(matches[i])
            if matchScore > 0:
                self.state[2*NUM_ENTITIES+i] = 1
            else:
                self.state[3*NUM_ENTITIES+i] = 1

            # self.state[2*NUM_ENTITIES+i] = float(matches[i])*confidences[i] if float(matches[i])>0 else -1*confidences[i]
        if nextArticle:
            # print self.indx, listNum, articleIndx
            # print COSINE_SIM[self.indx][listNum]
            self.state[4*NUM_ENTITIES] = self.articleSim(self.indx, listNum, articleIndx)
        else:
            self.state[4*NUM_ENTITIES] = 0

        #selectively mask states
        if self.entity != NUM_ENTITIES:
            for j in range(NUM_ENTITIES):
                if j != self.entity:
                    self.state[j] = 0
                    self.state[NUM_ENTITIES+j] = 0
                    #TODO: mask matches

        #add in context information
        if nextArticle and CONTEXT_TYPE != 0:
            j = 4*NUM_ENTITIES+1     
            for q in range(NUM_ENTITIES):
                if self.entity == NUM_ENTITIES or self.entity == q:
                    self.state[j:j+2*CONTEXT_LENGTH] = CONTEXT[self.indx][listNum][articleIndx+1][q]
                j += 2*CONTEXT_LENGTH

        # pdb.set_trace()

        #update state variables
        self.prevEntities = entities
        self.prevConfidences = confidences
        self.prevListNum = listNum
        self.prevArticleIndx = articleIndx

        return

    # check if two entities are equal. Need to handle city
    def checkEquality(self, e1, e2):
        # if gold is unknown, then dont count that
        return e2!='' and (COUNT_ZERO or e2 != 'zero')  and e1.lower() == e2.lower()

    def checkEqualityShooter(self, e1, e2):
        if e2 == '' or e2=='unknown': return 0.

        gold = set(splitBars(e2.lower()))
        pred = set(splitBars(e1.lower()))
        correct = len(gold.intersection(pred))
        prec = float(correct)/len(pred)
        rec = float(correct)/len(gold)

        if self.shooterLenientEval:
            if correct > 0:
                return 1.
            else:
                return 0.
        else:
            if prec+rec > 0:
                f1 = (2*prec*rec)/(prec+rec)
            else:
                f1 = 0.
            return f1

    def checkEqualityCity(self, e1, e2):
        return e2!='' and e1.lower() == e2.lower()


    def calculateReward(self, oldEntities, newEntities):
        if constants.mode == 'Shooter':
            rewards = [int(self.checkEquality(newEntities[1], self.goldEntities[1])) - int(self.checkEquality(oldEntities[1], self.goldEntities[1])),
                        int(self.checkEquality(newEntities[2], self.goldEntities[2])) - int(self.checkEquality(oldEntities[2], self.goldEntities[2]))]


            #add in shooter reward
            if self.goldEntities[0]:
                rewards.insert(0, self.checkEqualityShooter(newEntities[0], self.goldEntities[0]) \
                        - self.checkEqualityShooter(oldEntities[0], self.goldEntities[0]))
            else:
                rewards.insert(0, 0.)

            # add in city reward
            rewards.append(self.checkEqualityCity(newEntities[-1], self.goldEntities[-1]) \
                    - self.checkEqualityCity(oldEntities[-1], self.goldEntities[-1]))
        else:
            rewards = []
            for i in range(len(newEntities)):
                if self.goldEntities[i] != 'unknown':
                    rewards.append(self.checkEqualityShooter(newEntities[i], self.goldEntities[i]) - self.checkEqualityShooter(oldEntities[i], self.goldEntities[i]))
                else:
                    rewards.append(0.)



        if self.entity == NUM_ENTITIES:            
            return sum(rewards)
        else:
            return rewards[self.entity]


    def calculateStatSign(self, oldEntities, newEntities):
        if constants.mode == 'Shooter':
            rewards = [int(self.checkEquality(newEntities[1], self.goldEntities[1])) - int(self.checkEquality(oldEntities[1], self.goldEntities[1])),
                        int(self.checkEquality(newEntities[2], self.goldEntities[2])) - int(self.checkEquality(oldEntities[2], self.goldEntities[2]))]


            #add in shooter reward
            if self.goldEntities[0]:
                rewards.insert(0, self.checkEqualityShooter(newEntities[0], self.goldEntities[0]) \
                        - self.checkEqualityShooter(oldEntities[0], self.goldEntities[0]))
            else:
                rewards.insert(0, 0.)

            # add in city reward
            rewards.append(self.checkEqualityCity(newEntities[-1], self.goldEntities[-1]) \
                    - self.checkEqualityCity(oldEntities[-1], self.goldEntities[-1]))
        else:
            rewards = []
            for i in range(len(newEntities)):
                if self.goldEntities[i] != 'unknown':
                    rewards.append(self.checkEqualityShooter(newEntities[i], self.goldEntities[i]) - self.checkEqualityShooter(oldEntities[i], self.goldEntities[i]))
                else:
                    rewards.append(0.)

        return rewards
        


    #evaluate the bestEntities retrieved so far for a single article
    #IMP: make sure the evaluate variables are properly re-initialized
    def evaluateArticle(self, predEntities, goldEntities, shooterLenientEval, shooterLastName, evalOutFile):
        # print "Evaluating article", predEntities, goldEntities

        if constants.mode == 'Shooter':
            #shooterName first: only add this if gold contains a valid shooter
            if goldEntities[0]!='':
                if shooterLastName:
                    gold = set(splitBars(goldEntities[0].lower())[-1:])
                else:
                    gold = set(splitBars(goldEntities[0].lower()))

                pred = set(splitBars(predEntities[0].lower()))
                correct = len(gold.intersection(pred))

                if shooterLenientEval:
                    CORRECT[int2tags[0]] += (1 if correct> 0 else 0)
                    GOLD[int2tags[0]] += (1 if len(gold) > 0 else 0)
                    PRED[int2tags[0]] += (1 if len(pred) > 0 else 0)
                else:
                    CORRECT[int2tags[0]] += correct
                    GOLD[int2tags[0]] += len(gold)
                    PRED[int2tags[0]] += len(pred)



            #all other tags
            for i in range(1, NUM_ENTITIES):
                if COUNT_ZERO or goldEntities[i] != 'zero':
                    # gold = set(goldEntities[i].lower().split())
                    # pred = set(predEntities[i].lower().split())
                    # correct = len(gold.intersection(pred))
                    # GOLD[int2tags[i]] += len(gold)
                    # PRED[int2tags[i]] += len(pred)
                    GOLD[int2tags[i]] += 1
                    PRED[int2tags[i]] += 1
                    if predEntities[i].lower() == goldEntities[i].lower():
                        CORRECT[int2tags[i]] += 1
        else:
            #all other tags
            for i in range(NUM_ENTITIES):
                if goldEntities[i] != 'unknown':
                    
                    #old eval
                    gold = set(splitBars(goldEntities[i].lower()))
                    pred = set(splitBars(predEntities[i].lower()))
                    # if 'unknown' in pred:
                        # pred = set()                    
                    correct = len(gold.intersection(pred))

                    if shooterLenientEval:
                        CORRECT[int2tags[i]] += (1 if correct> 0 else 0)
                        GOLD[int2tags[i]] += (1 if len(gold) > 0 else 0)
                        PRED[int2tags[i]] += (1 if len(pred) > 0 else 0)
                    else:
                        CORRECT[int2tags[i]] += correct
                        GOLD[int2tags[i]] += len(gold)
                        PRED[int2tags[i]] += len(pred)

                    # print i, pred, "###", gold, "$$$", correct                    

                    #new eval (Adam)
                    # pred = predEntities[i].lower()
                    # gold = goldEntities[i].lower()
                    # if pred in gold:
                    #     CORRECT[int2tags[i]] += 1
                    # GOLD[int2tags[i]] += 1
                    # if pred != 'unknown':
                    #     PRED[int2tags[i]] += 1

                    
        if evalOutFile:
            evalOutFile.write("--------------------\n")
            evalOutFile.write("Gold: "+str(gold)+"\n")
            evalOutFile.write("Pred: "+str(pred)+"\n")
            evalOutFile.write("Correct: "+str(correct)+"\n")

    #TODO:use recall output for now. change this output later. 
    def oracleEvaluate(self, goldEntities, entityDic, confDic):
        # the best possible numbers assuming that just the right information is extracted
        # from the set of related articles
        global PRED, GOLD, CORRECT, EVALCONF, EVALCONF2
        bestPred, bestCorrect = collections.defaultdict(lambda:0.), collections.defaultdict(lambda:0.)
        bestConf = collections.defaultdict(lambda:0.)

        for stepNum, predEntitiesDic in entityDic.items():   
            for listNum, predEntities in predEntitiesDic.items():

                if constants.mode == 'Shooter':

                    #shooterName first: only add this if gold contains a valid shooter
                    if goldEntities[0]!='':
                        gold = set(splitBars(goldEntities[0].lower()))

                        pred = set(splitBars(predEntities[0].lower()))
                        correct = 1. if len(gold.intersection(pred)) > 0 else 0

                        if correct > bestCorrect[int2tags[0]] or (correct == bestCorrect[int2tags[0]] and len(pred) < bestPred[int2tags[0]]):
                            # print "Correct: ", correct
                            # print "Gold:", gold
                            # print "pred:", pred
                            bestCorrect[int2tags[0]] = correct
                            bestPred[int2tags[0]] = 1 if len(pred) > 0 else 0.
                            bestConf[int2tags[0]] = confDic[stepNum][listNum][0]

                        if stepNum == 0 and listNum == 0:
                            GOLD[int2tags[0]] += (1 if len(gold) > 0 else 0)

                        if correct==0:
                            EVALCONF2[int2tags[0]].append(confDic[stepNum][listNum][0])


                    #all other tags
                    for i in range(1, NUM_ENTITIES): 
                        if not COUNT_ZERO and goldEntities[i].lower() == 'zero': continue  
                        gold = set(goldEntities[i].lower().split())
                        pred = set(predEntities[i].lower().split())
                        correct = len(gold.intersection(pred))      
                        if correct > bestCorrect[int2tags[i]] or (correct == bestCorrect[int2tags[i]] and len(pred) < bestPred[int2tags[i]]):
                            bestCorrect[int2tags[i]] = correct
                            bestPred[int2tags[i]] = len(pred)
                            bestConf[int2tags[i]] = confDic[stepNum][listNum][i]                        
                        if stepNum == 0 and listNum == 0:
                            GOLD[int2tags[i]] += len(gold)
                        if correct==0:
                            EVALCONF2[int2tags[i]].append(confDic[stepNum][listNum][i])
                else:
                    #EMA
                    for i in range(NUM_ENTITIES): 
                        if goldEntities[i].lower() == 'unknown': continue  
                        gold = set(splitBars(goldEntities[i].lower()))
                        pred = set(splitBars(predEntities[i].lower()))                  
                        correct = 1. if len(gold.intersection(pred)) > 0 else 0

                        if correct > bestCorrect[int2tags[i]] or (correct == bestCorrect[int2tags[i]] and len(pred) < bestPred[int2tags[i]]):
                            bestCorrect[int2tags[i]] = correct
                            bestPred[int2tags[i]] = 1 if len(pred) > 0 else 0
                            # bestPred[int2tags[i]] = 1 if 'unknown' not in pred else bestPred[int2tags[i]]
                            bestConf[int2tags[i]] = confDic[stepNum][listNum][i]
                            # print "Correct: ", correct
                            # print "Gold:", gold
                            # print "pred:", pred
                        if stepNum == 0 and listNum == 0:
                            GOLD[int2tags[i]] += 1 #len(gold)
                        if correct==0:
                            EVALCONF2[int2tags[i]].append(confDic[stepNum][listNum][i])

        for i in range(NUM_ENTITIES):
            PRED[int2tags[i]] += bestPred[int2tags[i]]
            CORRECT[int2tags[i]] += bestCorrect[int2tags[i]]
            EVALCONF[int2tags[i]].append(bestConf[int2tags[i]])

    #TODO for EMA
    def thresholdEvaluate(self, goldEntities, args):
        #use a tf-idf threshold to select articles to extract from
        # uses Majority aggregation
        global PRED, GOLD, CORRECT, EVALCONF, EVALCONF2
        global ENTITIES, CONFIDENCES
        thres = args.threshold
        bestPred, bestCorrect = collections.defaultdict(lambda:0.), collections.defaultdict(lambda:0.)
        bestConf = collections.defaultdict(lambda:0.)
        bestSim = 0.
        bestEntities = ['','','','']
        aggEntites = collections.defaultdict(lambda:collections.defaultdict(lambda:0.))

        #add in the original entities
        for i in range(NUM_ENTITIES):
            aggEntites[i][self.bestEntities[i]] += 1.1


        for listNum in range(len(self.newArticles)):
            for i in range(len(self.newArticles[listNum])):
                sim = self.articleSim(self.indx, listNum, i)
                # print sim

                # if sim > bestSim:
                #     bestSim = sim
                #     entities, confidences = self.extractEntitiesWithConfidences(self.newArticles[i])
                #     bestEntities = entities
                #     bestConf = confidences
                #     print bestSim
                if sim > thres:
                    if (i+1) in ENTITIES[self.indx][listNum]:
                        entities, confidences = ENTITIES[self.indx][listNum][i+1], CONFIDENCES[self.indx][listNum][i+1]
                    else:
                        entities, confidences = self.extractEntitiesWithConfidences(self.newArticles[listNum][i])
                    for j in range(NUM_ENTITIES):
                        if entities[j] and entities[j] != 'unknown':
                            aggEntites[j][entities[j]] += 1


        #choose the best entities now
        for i in range(NUM_ENTITIES):
            tmp = sorted(aggEntites[i].items(), key=itemgetter(1), reverse=True)
            bestEntities[i] = tmp[0][0]

            print i, tmp
            # pdb.set_trace()


        self.evaluateArticle(bestEntities, goldEntities, args.shooterLenientEval, args.shooterLastName, args.evalOutFile)


    #TODO for EMA
    def confEvaluate(self, goldEntities, args):
        # use confidence-based aggregation over tf-idf similarity threshold used to select articles
        global PRED, GOLD, CORRECT, EVALCONF, EVALCONF2
        global ENTITIES, CONFIDENCES
        thres = args.threshold
        bestPred, bestCorrect = collections.defaultdict(lambda:0.), collections.defaultdict(lambda:0.)
        bestConf = collections.defaultdict(lambda:0.)
        bestSim = 0.
        bestEntities = ['','','','']
        bestConfidences = [0.,0.,0.,0.]
        aggEntites = collections.defaultdict(lambda:collections.defaultdict(lambda:0.))

        #add in the original entities
        for i in range(NUM_ENTITIES):
            if self.bestConfidences[i] > bestConfidences[i]:
                bestConfidences[i] = self.bestConfidences[i]
                bestEntities[i] = self.bestEntities[i]


        for listNum in range(len(self.newArticles)):
            for i in range(len(self.newArticles[listNum])):
                sim = self.articleSim(self.indx, listNum, i)
                if sim <= thres: continue
                if (i+1) in ENTITIES[self.indx][listNum]:
                    entities, confidences = ENTITIES[self.indx][listNum][i+1], CONFIDENCES[self.indx][listNum][i+1]
                else:
                    entities, confidences = self.extractEntitiesWithConfidences(self.newArticles[listNum][i])
                for j in range(NUM_ENTITIES):
                    if entities[j] != '' and entities[j] != 'unknown':
                        if confidences[j] > bestConfidences[j]:
                            bestConfidences[j] = confidences[j]
                            bestEntities[j] = entities[j]



        self.evaluateArticle(bestEntities, goldEntities, args.shooterLenientEval, args.shooterLastName, args.evalOutFile)

    #TODO for EMA
    #TODO: use conf or 1 for mode calculation
    def majorityVote(self, entityList):
        if not entityList: return '',0.

        dic = collections.defaultdict(lambda:0.)
        confDic = collections.defaultdict(lambda:0.)
        cnt = collections.defaultdict(lambda:0.)
        ticker = 0
        for entity, conf in entityList:
            dic[entity] += 1
            cnt[entity] += 1
            confDic[entity] += conf
            if ticker == 0: dic[entity] += 0.1 #extra for original article to break ties
            ticker += 1

        bestEntity, bestVote = sorted(dic.items(), key=itemgetter(1), reverse=True)[0]
        return bestEntity, confDic[bestEntity]/cnt[bestEntity]


    #take a single step in the episode
    def step(self, action, query):
        global CHANGES
        oldEntities = copy.copy(self.bestEntities.values())

        #update pointer to next article
        listNum = query-1
        self.stepNum[listNum] += 1
    
        self.updateState(action, query, self.ignoreDuplicates)

        newEntities = self.bestEntities.values()


        if self.delayedReward == 'True':
            reward = self.calculateReward(self.originalEntities, newEntities)
        else:
            reward = self.calculateReward(oldEntities, newEntities)

        # negative per step
        reward -= 0.001

        return self.state, reward, self.terminal


def loadFile(filename):
    articles, titles, identifiers, downloaded_articles = [], [] ,[] ,[]

    #load data and process identifiers
    with open(filename, "rb") as inFile:
        while True:
            try:
                a, b, c, d = pickle.load(inFile)
                articles.append(a)
                titles.append(b)
                identifiers.append(c)
                downloaded_articles.append(d)
            except:
                break

    identifiers_tmp = []
    for e in identifiers:
        for i in range(NUM_ENTITIES):       
            if type(e[i]) == int or e[i].isdigit():
                e[i] = int(e[i])
                e[i] = inflect_engine.number_to_words(e[i])
       
        identifiers_tmp.append(e)
    identifiers = identifiers_tmp

    return articles, titles, identifiers, downloaded_articles


def baselineEval(articles, identifiers, args):
    global CORRECT, GOLD, PRED
    CORRECT = collections.defaultdict(lambda:0.)
    GOLD = collections.defaultdict(lambda:0.)
    PRED = collections.defaultdict(lambda:0.)
    for indx in range(len(articles)):
        print "INDX:", indx, '/', len(articles)
        originalArticle = articles[indx][0] #since article has words and tags
        newArticles = [[] for i in range(NUM_QUERY_TYPES)]
        goldEntities = identifiers[indx]
        env = Environment(originalArticle, newArticles, goldEntities, indx, args, evalMode=True)
        env.evaluateArticle(env.bestEntities.values(), env.goldEntities, args.shooterLenientEval, args.shooterLastName, args.evalOutFile)

    print "------------\nEvaluation Stats: (Precision, Recall, F1):"
    for tag in int2tags:
        prec = CORRECT[tag]/PRED[tag]
        rec = CORRECT[tag]/GOLD[tag]
        f1 = (2*prec*rec)/(prec+rec)
        print tag, prec, rec, f1, "########", CORRECT[tag], PRED[tag], GOLD[tag]


def thresholdEval(articles, downloaded_articles, identifiers, args):
    global CORRECT, GOLD, PRED
    CORRECT = collections.defaultdict(lambda:0.)
    GOLD = collections.defaultdict(lambda:0.)
    PRED = collections.defaultdict(lambda:0.)
    for indx in range(len(articles)):
        print "INDX:", indx
        originalArticle = articles[indx][0]
        newArticles = [[q.split(' ')[:WORD_LIMIT] for q in sublist] for sublist in downloaded_articles[indx]]
        goldEntities = identifiers[indx]
        env = Environment(originalArticle, newArticles, goldEntities, indx, args, False)
        env.thresholdEvaluate(env.goldEntities, args)

    print "------------\nEvaluation Stats: (Precision, Recall, F1):"
    for tag in int2tags:
        prec = CORRECT[tag]/PRED[tag]
        rec = CORRECT[tag]/GOLD[tag]
        f1 = (2*prec*rec)/(prec+rec)
        print tag, prec, rec, f1

def confEval(articles, downloaded_articles, identifiers, args):
    global CORRECT, GOLD, PRED
    CORRECT = collections.defaultdict(lambda:0.)
    GOLD = collections.defaultdict(lambda:0.)
    PRED = collections.defaultdict(lambda:0.)
    for indx in range(len(articles)):
        print "INDX:", indx
        originalArticle = articles[indx][0]
        newArticles = [[q.split(' ')[:WORD_LIMIT] for q in sublist] for sublist in downloaded_articles[indx]]
        goldEntities = identifiers[indx]
        env = Environment(originalArticle, newArticles, goldEntities, indx, args, False)
        env.confEvaluate(env.goldEntities, args)

    print "------------\nEvaluation Stats: (Precision, Recall, F1):"
    for tag in int2tags:
        prec = CORRECT[tag]/PRED[tag]
        rec = CORRECT[tag]/GOLD[tag]
        if prec+rec > 0:
            f1 = (2*prec*rec)/(prec+rec)
        else:
            f1 = 0
        print tag, prec, rec, f1

def plot_hist(evalconf, name):
    for i in evalconf.keys():
        plt.hist(evalconf[i], bins=[0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.title("Gaussian Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(name+"_"+str(i)+".png")
        plt.clf()

def computeContext(ENTITIES, CONTEXT, ARTICLES, DOWNLOADED_ARTICLES, vectorizer, context=3):
    # calculate the context variables for all articles
    print "Computing context..."

    vocab = vectorizer.vocabulary_

    for indx, lists in ENTITIES.items():
        print '\r', indx, '/', len(ENTITIES),
        for listNum, articles in lists.items():
            for articleNum, entities in articles.items():
                CONTEXT[indx][listNum][articleNum] = []
                if articleNum == 0:
                    #original article
                    article = [w.lower() for w in ARTICLES[indx][0]] #need only the tokens, not tags
                else:
                    raw_article = DOWNLOADED_ARTICLES[indx][listNum][articleNum-1].lower()
                    cleaned_article = re.sub(r'[^\x00-\x7F]+',' ', raw_article)
                    article = word_tokenize(cleaned_article)
                for entityNum, entity in enumerate(entities):
                    vec = []
                    phrase = []

                    if entityNum == 0 or constants.mode == 'EMA': #shooter case or EMA mode
                        entity = set(splitBars(entity))
                        for i, word in enumerate(article):
                            if word in entity:
                                for j in range(1,context+1):
                                    if i-j>=0:
                                        phrase.append(article[i-j])
                                    else:
                                        phrase.append('XYZUNK') #random unseen phrase
                                for j in range(1,context+1):
                                    if i+j < len(article):
                                        phrase.append(article[i+j])
                                    else:
                                        phrase.append('XYZUNK') #random unseen phrase
                                break
                        mat = vectorizer.transform([' '.join(phrase)]).toarray()

                        for w in phrase:
                            feat_indx = vocab.get(w)
                            if feat_indx:
                                vec.append(float(mat[0,feat_indx]))
                            else:
                                vec.append(0.)
                    else:
                        for i, word in enumerate(article):
                            if type(word) == int or word.isdigit() and word < 100:
                                word = int(word)
                                word = inflect_engine.number_to_words(word)
                            if word in entity:
                                for j in range(1,context+1):
                                    if i-j>=0:
                                        phrase.append(article[i-j])
                                    else:
                                        phrase.append('XYZUNK') #random unseen phrase
                                for j in range(1,context+1):
                                    if i+j < len(article):
                                        phrase.append(article[i+j])
                                    else:
                                        phrase.append('XYZUNK') #random unseen phrase
                                break

                        mat = vectorizer.transform([' '.join(phrase)]).toarray()
                        for w in phrase:
                            feat_indx = vocab.get(w)
                            if feat_indx:
                                vec.append(float(mat[0,feat_indx]))
                            else:
                                vec.append(0.)

                    # take care of all corner cases
                    if len(vec) == 0:
                        vec = [0. for q in range(2*context)]

                    #now store the vector
                    CONTEXT[indx][listNum][articleNum].append(vec)
                    try:
                        assert(len(CONTEXT[indx][listNum][articleNum]) <= NUM_ENTITIES)
                    except:
                        pdb.set_trace()


    print "done."
    return CONTEXT

def splitDict(dict, start, end):
        return [dict[i] for i in range(start, end)]

def main(args):
    global ENTITIES, CONFIDENCES, COSINE_SIM, CONTEXT
    global TRAIN_ENTITIES, TRAIN_CONFIDENCES, TRAIN_COSINE_SIM, TRAIN_CONTEXT
    global TEST_ENTITIES, TEST_CONFIDENCES, TEST_COSINE_SIM, TEST_CONTEXT
    global evalMode
    global CORRECT, GOLD, PRED, EVALCONF, EVALCONF2
    global QUERY, ACTION, CHANGES
    global trained_model
    global CONTEXT_TYPE
    global STAT_POSITIVE, STAT_NEGATIVE
    
    print args

    trained_model = pickle.load( open(args.modelFile, "rb" ) )

    #load cached entities (speed up)
    train_articles, train_titles, train_identifiers, train_downloaded_articles, TRAIN_ENTITIES, TRAIN_CONFIDENCES, TRAIN_COSINE_SIM, CONTEXT1, CONTEXT2 = pickle.load(open(args.trainEntities, "rb"))

    if args.contextType == 1:
        TRAIN_CONTEXT = CONTEXT1
    else:
        TRAIN_CONTEXT = CONTEXT2

    CONTEXT_TYPE = args.contextType

    
    test_articles, test_titles, test_identifiers, test_downloaded_articles, TEST_ENTITIES, TEST_CONFIDENCES, TEST_COSINE_SIM, CONTEXT1, CONTEXT2 = pickle.load(open(args.testEntities, "rb"))

    if args.contextType == 1:
        TEST_CONTEXT = CONTEXT1
    else:
        TEST_CONTEXT = CONTEXT2

    print len(train_articles)
    print len(test_articles)


    #starting assignments
    if not args.baselineEval and not args.thresholdEval and not args.confEval:
        ENTITIES = TRAIN_ENTITIES
        CONFIDENCES = TRAIN_CONFIDENCES
        COSINE_SIM = TRAIN_COSINE_SIM
        CONTEXT = TRAIN_CONTEXT
        articles, titles, identifiers, downloaded_articles = train_articles, train_titles, train_identifiers, train_downloaded_articles
    else:
        ENTITIES = TEST_ENTITIES
        CONFIDENCES = TEST_CONFIDENCES
        COSINE_SIM = TEST_COSINE_SIM
        CONTEXT = TEST_CONTEXT
        articles, titles, identifiers, downloaded_articles = test_articles, test_titles, test_identifiers, test_downloaded_articles

    if args.baselineEval:        
        baselineEval(articles, identifiers, args)
        return
    elif args.thresholdEval:
        thresholdEval(articles, downloaded_articles, identifiers, args)
        return
    elif args.confEval:
        confEval(articles, downloaded_articles, identifiers, args)
        return
    elif args.classifierEval:
        print args.trainEntities
        print args.testEntities


        m = "TEST"
        split_index = 292
        if  m == "DEV":
            CLS_TEST_ENTITIES = splitDict(TRAIN_ENTITIES, split_index, len(TRAIN_ENTITIES) )
            CLS_TEST_CONFIDENCES = splitDict(TRAIN_CONFIDENCES, split_index, len(TRAIN_ENTITIES) )
            CLS_TEST_COSINE_SIM = splitDict(TRAIN_COSINE_SIM, split_index, len(TRAIN_ENTITIES) )
            CLS_TEST_CONTEXT = splitDict(TRAIN_CONTEXT, split_index, len(TRAIN_ENTITIES) )
            CLS_test_identifiers = splitDict(train_identifiers,  split_index, len(TRAIN_ENTITIES) )
                     
            CLS_TRAIN_ENTITIES = splitDict(TRAIN_ENTITIES, 0, split_index )
            CLS_TRAIN_CONFIDENCES = splitDict(TRAIN_CONFIDENCES, 0, split_index )
            CLS_TRAIN_COSINE_SIM = splitDict(TRAIN_COSINE_SIM, 0, split_index )
            CLS_TRAIN_CONTEXT= splitDict(TRAIN_CONTEXT, 0, split_index )

            CLS_train_identifiers   = splitDict(train_identifiers, 0, split_index )
        elif m == "TEST":
            CLS_TRAIN_ENTITIES =TRAIN_ENTITIES 
            CLS_TRAIN_CONFIDENCES =TRAIN_CONFIDENCES 
            CLS_TRAIN_COSINE_SIM =TRAIN_COSINE_SIM 
            CLS_TRAIN_CONTEXT =TRAIN_CONTEXT 
                     
            CLS_TEST_ENTITIES = TEST_ENTITIES 
            CLS_TEST_CONFIDENCES = TEST_CONFIDENCES 
            CLS_TEST_COSINE_SIM = TEST_COSINE_SIM 
            CLS_TEST_CONTEXT=     TEST_CONTEXT

       
            CLS_train_identifiers = train_identifiers
            CLS_test_identifiers   = test_identifiers

        baseline = Classifier(CLS_TRAIN_ENTITIES, CLS_TRAIN_CONFIDENCES, CLS_TRAIN_COSINE_SIM, CLS_TRAIN_CONTEXT,\
                 CLS_TEST_ENTITIES, CLS_TEST_CONFIDENCES, CLS_TEST_COSINE_SIM, CLS_TEST_CONTEXT)
        
        baseline.trainAndEval(CLS_train_identifiers, CLS_test_identifiers, args.entity, COUNT_ZERO)
        return


    articleNum = 0
    savedArticleNum = 0

    outFile = open(args.outFile, 'w', 0) #unbuffered
    outFile.write(str(args)+"\n")

    outFile2 = open(args.outFile+'.2', 'w', 0) #for analysis
    outFile2.write(str(args)+"\n")



    evalOutFile = None
    if args.evalOutFile != '':
        evalOutFile = open(args.evalOutFile, 'w')

    # pdb.set_trace()

    #server setup
    port = args.port
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    print "Started server on port", port

    #for analysis
    stepCnt = 0

    # server loop
    while True:
        #  Wait for next request from client
        message = socket.recv()
        # print "Received request: ", message

        if message == "newGame":
            # indx = articleNum % 10 #for test
            indx = articleNum % len(articles)
            if DEBUG: print "INDX:", indx
            articleNum += 1
            originalArticle = articles[indx][0] #since article has words and tags
            #IMP: make sure downloaded_articles is of form <indx, listNum>
            newArticles = [[q.split(' ')[:WORD_LIMIT] for q in sublist] for sublist in downloaded_articles[indx]]
            goldEntities = identifiers[indx]
            env = Environment(originalArticle, newArticles, goldEntities, indx, args, evalMode)
            newstate, reward, terminal = env.state, 0, 'false'

        elif message == "evalStart":
            CORRECT = collections.defaultdict(lambda:0.)
            GOLD = collections.defaultdict(lambda:0.)
            PRED = collections.defaultdict(lambda:0.)
            QUERY = collections.defaultdict(lambda:0.)
            ACTION = collections.defaultdict(lambda:0.)
            CHANGES = 0
            evalMode = True
            savedArticleNum = articleNum
            articleNum = 0
            stepCnt = 0
            STAT_POSITIVE, STAT_NEGATIVE = [0 for i in range(NUM_ENTITIES)], [0 for i in range(NUM_ENTITIES)]

            ENTITIES = TEST_ENTITIES
            CONFIDENCES = TEST_CONFIDENCES
            COSINE_SIM = TEST_COSINE_SIM
            CONTEXT = TEST_CONTEXT
            articles, titles, identifiers, downloaded_articles = test_articles, test_titles, test_identifiers, test_downloaded_articles

            # print "##### Evaluation Started ######"

        elif message == "evalEnd":
            print "------------\nEvaluation Stats: (Precision, Recall, F1):"
            outFile.write("------------\nEvaluation Stats: (Precision, Recall, F1):\n")
            for tag in int2tags:
                prec = CORRECT[tag]/PRED[tag]
                rec = CORRECT[tag]/GOLD[tag]
                f1 = (2*prec*rec)/(prec+rec)
                print tag, prec, rec, f1, "########", CORRECT[tag], PRED[tag], GOLD[tag]
                outFile.write(' '.join([str(tag), str(prec), str(rec), str(f1)])+'\n')
            print "StepCnt (total, average):", stepCnt, float(stepCnt)/len(articles)
            outFile.write("StepCnt (total, average): " + str(stepCnt)+ ' ' + str(float(stepCnt)/len(articles)) + '\n')
            

            qsum = sum(QUERY.values())
            asum = sum(ACTION.values())
            outFile2.write("------------\nQsum: " + str(qsum) +  " Asum: " +  str(asum)+'\n')
            for k, val in QUERY.items():
                outFile2.write("Query " + str(k) + ' ' + str(val/qsum)+'\n')
            for k, val in ACTION.items():
                outFile2.write("Action " + str(k) + ' ' + str(val/asum)+'\n')
            outFile2.write("CHANGES: "+str(CHANGES)+ ' ' + str(float(CHANGES)/len(articles))+"\n")
            outFile2.write("STAT_POSITIVE, STAT_NEGATIVE "+str(STAT_POSITIVE) + ', ' +str(STAT_NEGATIVE)+'\n')

            #for analysis
            # pdb.set_trace()

            evalMode = False            
            articleNum = savedArticleNum

            ENTITIES = TRAIN_ENTITIES
            CONFIDENCES = TRAIN_CONFIDENCES
            COSINE_SIM = TRAIN_COSINE_SIM
            CONTEXT = TRAIN_CONTEXT
            articles, titles, identifiers, downloaded_articles = train_articles, train_titles, train_identifiers, train_downloaded_articles
            # print "##### Evaluation Ended ######"


            if args.oracle:
                plot_hist(EVALCONF, "conf1")
                plot_hist(EVALCONF2, "conf2")

            #save the extracted entities
            if args.saveEntities:
                pickle.dump([TRAIN_ENTITIES, TRAIN_CONFIDENCES, TRAIN_COSINE_SIM], open("train2.entities", "wb"))
                pickle.dump([TEST_ENTITIES, TEST_CONFIDENCES, TEST_COSINE_SIM], open("test2.entities", "wb"))
                return

        else:
            # message is "step"
            action, query = [int(q) for q in message.split()]

            if evalMode:
                ACTION[action] += 1
                QUERY[query] += 1

            if evalMode and DEBUG:
                print "State:"
                print newstate[:4]
                print newstate[4:8]
                print newstate[8:]
                print "Entities:", env.prevEntities
                print "Action:", action, query

            newstate, reward, terminal = env.step(action, query)
            terminal = 'true' if terminal else 'false'

            #remove reward unless terminal
            if args.delayedReward == 'True' and terminal == 'false':
                reward = 0

            if evalMode and DEBUG and reward != 0:
                print "Reward:", reward
                pdb.set_trace()

        if message != "evalStart" and message != "evalEnd":
            #do article eval if terminal
            if evalMode and articleNum <= len(articles) and terminal == 'true':
                if args.oracle:                    
                    env.oracleEvaluate(env.goldEntities, ENTITIES[env.indx], CONFIDENCES[env.indx])
                else:
                    env.evaluateArticle(env.bestEntities.values(), env.goldEntities, args.shooterLenientEval, args.shooterLastName, evalOutFile)

                stepCnt += sum(env.stepNum)

                #stat sign
                vals = env.calculateStatSign(env.originalEntities, env.bestEntities.values())
                for i, val in enumerate(vals):
                    if val > 0:
                        STAT_POSITIVE[i] += val
                    else:
                        STAT_NEGATIVE[i] -= val

                #for analysis
                for entityNum in [0,1,2,3]:
                    if ANALYSIS and evalMode and env.bestEntities.values()[entityNum].lower() != env.originalEntities[entityNum].lower() and reward > 0:
                        CHANGES += 1
                        try:
                            print "ENTITY:", entityNum
                            print "Entities:", 'best', env.bestEntities.values()[entityNum], 'orig', env.originalEntities[entityNum], 'gold', env.goldEntities[entityNum]
                            print ' '.join(originalArticle)
                            print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                            print ' '.join(newArticles[env.bestIndex[0]][env.bestIndex[1]])
                            print "----------------------------"
                        except:
                            pass

            #send message (IMP: only for newGame or step messages)
            outMsg = 'state, reward, terminal = ' + str(newstate) + ',' + str(reward)+','+terminal
            socket.send(outMsg.replace('[', '{').replace(']', '}'))
        else:
            socket.send("done")


if __name__ == '__main__':
    env = None
    newstate, reward, terminal = None, None, None

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--port",
        type = int,
        default = 5050,
        help = "port for server")
    argparser.add_argument("--trainFile",
        type = str,
        help = "training File")
    argparser.add_argument("--testFile",
        type = str,
        default = "",
        help = "Testing File")
    argparser.add_argument("--outFile",
        type = str,
        help = "Output File")

    argparser.add_argument("--evalOutFile",
        default = "",
        type = str,
        help = "Output File for predictions")

    argparser.add_argument("--modelFile",
        type = str,
        help = "Model File")

    argparser.add_argument("--shooterLenientEval",
        type = bool,
        default = False,
        help = "Evaluate shooter leniently by counting any match as right")

    argparser.add_argument("--shooterLastName",
        type = bool,
        default = False,
        help = "Evaluate shooter using only last name")

    argparser.add_argument("--oracle",
        type = bool,
        default = False,
        help = "Evaluate using oracle")

    argparser.add_argument("--ignoreDuplicates",
        type = bool,
        default = False,
        help = "Ignore duplicate articles in downloaded ones.")

    argparser.add_argument("--baselineEval",
        type = bool,
        default = False,
        help = "Evaluate baseline performance")

    argparser.add_argument("--classifierEval",
        type = bool,
        default = False,
        help = "Evaluate performance using a simple maxent classifier")

    argparser.add_argument("--thresholdEval",
        type = bool,
        default = False,
        help = "Use tf-idf similarity threshold to select articles to extract from")

    argparser.add_argument("--threshold",
        type = float,
        default = 0.8,
        help = "threshold value for Aggregation baseline above")

    argparser.add_argument("--confEval",
        type = bool,
        default = False,
        help = "Evaluate with best conf ")

    argparser.add_argument("--rlbasicEval",
        type = bool,
        default = False,
        help = "Evaluate with RL agent that takes only reconciliation decisions.")

    argparser.add_argument("--rlqueryEval",
        type = bool,
        default = False,
        help = "Evaluate with RL agent that takes only query decisions.")

    argparser.add_argument("--shuffleArticles",
        type = bool,
        default = False,
        help = "Shuffle the order of new articles presented to agent")

    argparser.add_argument("--entity",
        type = int,
        default = NUM_ENTITIES,
        help = "Entity num. 4 means all.")

    argparser.add_argument("--aggregate",
        type = str,
        default = 'always',
        help = "Options: always, conf, majority")

    argparser.add_argument("--delayedReward",
        type = str,
        default = 'False',
        help = "delay reward to end")

    argparser.add_argument("--trainEntities",
        type = str,
        default = '',
        help = "Pickle file with extracted train entities")

    argparser.add_argument("--testEntities",
        type = str,
        default = '',
        help = "Pickle file with extracted test entities")

    argparser.add_argument("--numEntityLists",
        type = int,
        default = 1,
        help = "number of different query lists to consider")

    argparser.add_argument("--contextType",
        type = int,
        default = 1,
        help = "Type of context to consider (1 = counts, 2 = tfidf, 0 = none)")    


    argparser.add_argument("--saveEntities",
        type = bool,
        default = False,
        help = "save extracted entities to file")


    args = argparser.parse_args()

    main(args)

#sample
#python server.py --port 7000 --trainEntities consolidated/train+context.5.p --testEntities consolidated/dev+test+context.5.p --outFile outputs/tmp2.out --modelFile trained_model2.p --entity 4 --aggregate always --shooterLenientEval True --delayedReward False --contextType 2


