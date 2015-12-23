import zmq, time
import numpy as np
import copy
import sys, json, pdb, pickle, operator, collections
import helper
import predict2 as predict
from train import load_data
from itertools import izip
import inflect
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


#Global variables
NUM_ENTITIES = 4
port = "5050"

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)
print "Started server on port", port

trained_model = pickle.load( open( "trained_model.p", "rb" ) )
tfidf_vectorizer = TfidfVectorizer()
inflect_engine = inflect.engine()

TFIDF_MATRICES = {}
ENTITIES = collections.defaultdict(lambda:{})
CONFIDENCES = collections.defaultdict(lambda:{})

#Environment for each episode
class Environment:

    def __init__(self, originalArticle, newArticles, goldEntities, indx):
        self.originalArticle = originalArticle
        self.newArticles = newArticles #extra articles to process
        self.goldEntities = goldEntities 
       
        #TODO: add slots for matches in values between DB and the new article
        self.state = [0 for i in range(2 * NUM_ENTITIES + 1)]
        self.terminal = False
        
        self.entities = collections.defaultdict(lambda:[]) #current Entities extracted
        self.bestEntities = collections.defaultdict(lambda:None) #current best entities
        self.bestConfidences = collections.defaultdict(lambda:0)

        # to keep track of extracted values from previousArticle
        if 0 in ENTITIES[indx]:
            self.prevEntities, self.prevConfidences = ENTITIES[indx][0], CONFIDENCES[indx][0]
        else:
            self.prevEntities, self.prevConfidences = self.extractEntitiesWithConfidences(self.originalArticle)
            ENTITIES[indx][0] = self.prevEntities
            CONFIDENCES[indx][0] = self.prevConfidences

        #calculate tf-idf similarities
        self.allArticles = [originalArticle] + self.newArticles
        self.allArticles = [' '.join(q) for q in self.allArticles]

        if indx in TFIDF_MATRICES:
            self.tfidf_matrix = TFIDF_MATRICES[indx]
        else:
            self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.allArticles)
            TFIDF_MATRICES[indx] = self.tfidf_matrix

        #update the initial state
        self.stepNum = 0
        self.updateState(1)

        # print "goldEntities", self.goldEntities
        # print "extracted", self.bestEntities
        # pdb.set_trace()

        return
    
    def extractEntitiesWithConfidences(self, article):
        #article is a list of words
        joined_article = ' '.join(article)
        pred, conf_scores, conf_cnts = predict.predictWithConfidences(trained_model, joined_article, False, helper.cities)

        for i in range(len(conf_scores)):
            if conf_cnts[i] > 0:
                conf_scores[i] /= conf_cnts[i]

        return pred.split(','), conf_scores

    #find the article similarity between original and newArticle[i] (=allArticles[i+1])
    def articleSim(self, i):
        return cosine_similarity(self.tfidf_matrix[0:1], self.tfidf_matrix[i+1:i+2])[0][0]

    # update the state based on the decision from DQN
    def updateState(self, action):

        #get next article
        if self.stepNum < len(self.newArticles):
            nextArticle = self.newArticles[self.stepNum]
        else:
            nextArticle = None

        if action == 1:
            # integrate the values into the current DB state
            entities, confidences = self.prevEntities, self.prevConfidences
            # print "best confidences", self.bestConfidences
            # print "new confidences", confidences
            for i in range(NUM_ENTITIES):
                self.entities[i].append(entities[i])
                if not self.bestEntities[i] or confidences[i] > self.bestConfidences[i]:
                    self.bestEntities[i] = entities[i]
                    self.bestConfidences[i] = confidences[i]
                    # print "Changing best Entities"
                    # print "New entities", self.bestEntities

        if nextArticle:    
            if (self.stepNum+1) in ENTITIES[indx]:
                entities, confidences = ENTITIES[indx][self.stepNum+1], CONFIDENCES[indx][self.stepNum+1]
            else:
                entities, confidences = self.extractEntitiesWithConfidences(nextArticle)
                ENTITIES[indx][self.stepNum+1], CONFIDENCES[indx][self.stepNum+1] = entities, confidences
            assert(len(entities) == len(confidences))          
        else:
            # print "No next article"
            entities, confidences = [""]*NUM_ENTITIES, [0]*NUM_ENTITIES
            self.terminal = True

        #modify self.state appropriately        
        for i in range(NUM_ENTITIES):
            self.state[i] = self.bestConfidences[i] #DB state
            self.state[NUM_ENTITIES+i] = confidences[i]  #next article state
            if nextArticle:
                self.state[-1] = self.articleSim(self.stepNum)
            else:
                self.state[-1] = 0

        #update state variables
        self.prevEntities = entities
        self.prevConfidences = confidences

        return

    # check if two entities are equal. Need to handle shooterName and city
    #TODO: handle the case when goldEntities does not have annotation
    def checkEquality(self, e1, e2):         
        # if gold is unknown, then dont count that
        return e2!=''  and e1.lower() == e2.lower()

    def calculateReward(self, oldEntities, newEntities):
        reward = sum(map(self.checkEquality, newEntities, self.goldEntities)) \
                - sum(map(self.checkEquality, oldEntities, self.goldEntities))

        #TODO: if terminal, give some reward based on how many entities are correct?

        return reward

    #take a single step in the episode
    def step(self, action):
        oldEntities = copy.copy(self.bestEntities.values())

        #update pointer to next article
        self.stepNum += 1
    

        self.updateState(action)

        newEntities = self.bestEntities.values()
        reward = self.calculateReward(oldEntities, newEntities)

        return self.state, reward, self.terminal

if __name__ == '__main__':
    env = None
    newstate, reward, terminal = None, None, None

    if len(sys.argv) > 1:
        trainFile = sys.argv[1]
    else:
        trainFile = 'train.extra'

    articles, titles, identifiers, downloaded_articles = [], [] ,[] ,[]

    #load data and process identifiers
    with open(trainFile, "rb") as inFile:
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
            try:
                e[i] = int(e[i])
                e[i] = inflect_engine.number_to_words(e[i])
            except:
                pass
        identifiers_tmp.append(e)
    identifiers = identifiers_tmp
        
    # pdb.set_trace()

    articleNum = 0

    # server loop
    while True:
        #  Wait for next request from client
        message = socket.recv()
        # print "Received request: ", message

        # UNCOMMENT THIS
        if message == "newGame":
            indx = articleNum % len(articles)
            print "INDX:", indx
            articleNum += 1
            originalArticle = articles[indx][0]
            newArticles = [q.split(' ')[:1000] for q in downloaded_articles[indx]]
            goldEntities = identifiers[indx]   
            env = Environment(originalArticle, newArticles, goldEntities, indx)
            newstate, reward, terminal = env.state, 0, 'false'
        else:
            action = int(message)
            newstate, reward, terminal = env.step(action)        
            terminal = 'true' if terminal else 'false'


        #send message
        outMsg = 'state, reward, terminal = ' + str(newstate) + ',' + str(reward)+','+terminal
        socket.send(outMsg.replace('[', '{').replace(']', '}'))

        
