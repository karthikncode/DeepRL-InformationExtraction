import zmq, time
import numpy as np
import copy
import sys, json, pdb, pickle, operator, collections
import helper
import predict2 as predict
from train import load_data
from itertools import izip

NUM_ENTITIES = 4

port = "5050"

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)
print "Started server on port", port

trained_model = pickle.load( open( "trained_model.p", "rb" ) )

#Environment for each episode
class Environment:

    def __init__(self, originalArticle, newArticles, goldEntities):
        self.originalArticle = originalArticle
        self.newArticles = newArticles #extra articles to process
        self.goldEntities = goldEntities 
        self.stepNum = 0
        #TODO: add slots for matches in values between DB and the new article
        self.state = [0 for i in range(2 * NUM_ENTITIES + 1)]
        self.terminal = False
        
        self.entities = collections.defaultdict(lambda:[]) #current Entities extracted
        self.bestEntities = collections.defaultdict(lambda:None) #current best entities
        self.bestConfidences = collections.defaultdict(lambda:0)

        # to keep track of extracted values from previousArticle
        self.prevEntities, self.prevConfidences = self.extractEntitiesWithConfidences(self.originalArticle)

        #update the initial state
        self.updateState(1, self.newArticles[0])

        return
    
    def extractEntitiesWithConfidences(self, article):
        #article is a list of words
        joined_article = ' '.join(article)
        pred, conf_scores, conf_cnts = predict.predictWithConfidences(trained_model, joined_article, False, helper.cities)

        for i in range(len(conf_scores)):
            if conf_cnts[i] > 0:
                conf_scores[i] /= conf_cnts[i]

        return pred.split(','), conf_scores

    #TODO: find the article similarity of a1 and a2
    def articleSim(self, a1, a2):
        return 0.5

    # update the state based on the decision from DQN
    def updateState(self, action, nextArticle):

        if action == 1:
            # integrate the values into the current DB state
            entities, confidences = self.prevEntities, self.prevConfidences
            for i in range(NUM_ENTITIES):
                self.entities[i].append(entities[i])
                if not self.bestEntities[i] or confidences[i] > self.bestConfidences[i]:
                    self.bestEntities[i] = entities[i]
                    self.bestConfidences[i] = confidences[i]

        if nextArticle:    
            entities, confidences = self.extractEntitiesWithConfidences(nextArticle)
            assert(len(entities) == len(confidences))          
        else:
            print "No next article"
            entities, confidences = [""]*NUM_ENTITIES, [0]*NUM_ENTITIES
            self.terminal = True

        #modify self.state appropriately        
        for i in range(NUM_ENTITIES):
            self.state[i] = self.bestConfidences[i] #DB state
            self.state[NUM_ENTITIES+i] = confidences[i]  #next article state
            self.state[-1] = self.articleSim(self.originalArticle, nextArticle)

        return

    # check if two entities are equal. Need to handle shooterName and city
    #TODO
    def checkEquality(self, e1, e2):
        return False

    def calculateReward(self, oldEntities, newEntities):
        reward = sum(map(self.checkEquality, newEntities, self.goldEntities)) \
                - sum(map(self.checkEquality, oldEntities, self.goldEntities))

        #TODO: if terminal, give some reward based on how many entities are correct?

        return reward

    #take a single step in the episode
    def step(self, action):
        oldEntities = copy.copy(self.bestEntities)

        #get next article
        self.stepNum += 1
        if self.stepNum < len(self.newArticles):
            nextArticle = self.newArticles[self.stepNum] #starts with 1
        else:
            nextArticle = None

        self.updateState(action, nextArticle)

        newEntities = self.bestEntities
        reward = self.calculateReward(oldEntities, newEntities)

        return self.state, reward, self.terminal

if __name__ == '__main__':
    env = None
    newstate, reward, terminal = None, None, None

    if len(sys.argv) > 1:
        trainFile = sys.argv[1]
    else:
        trainFile = '../data/tagged_data/whole_text_full_city/train.tag'

    articles, identifiers = load_data(trainFile)
    identifiers = [e.split(',')[:4] for e in identifiers]

    # server loop
    while True:
        #  Wait for next request from client
        message = socket.recv()
        print "Received request: ", message


        #TESTING - TODO: make this to reflect the correct set of articles
        originalArticle = articles[0][0]
        newArticles = [e[0] for e in articles[1:5]]
        goldEntities = identifiers[0]

        # UNCOMMENT THIS
        if message == "newGame":
            env = Environment(originalArticle, newArticles, goldEntities)
            newstate, reward, terminal = env.state, 0, 'false'
        else:
            action = int(message)
            newstate, reward, terminal = env.step(action)        
            terminal = 'true' if terminal else 'false'

        #send message
        outMsg = 'state, reward, terminal = ' + str(newstate) + ',' + str(reward)+','+terminal
        socket.send(outMsg.replace('[', '{').replace(']', '}'))

