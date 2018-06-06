from sklearn.linear_model import LogisticRegression as MaxEnt
import copy
import random
import collections
from itertools import izip
import constants
import helper
import warnings
from predict import evaluatePrediction
import pdb

warnings.filterwarnings("ignore")



class Classifier(object):

    def __init__(self, TRAIN_ENTITIES, TRAIN_CONFIDENCES, TRAIN_COSINE_SIM, TRAIN_CONTEXT,\
                 TEST_ENTITIES, TEST_CONFIDENCES, TEST_COSINE_SIM, TEST_CONTEXT):
    
        self.TRAIN_ENTITIES = TRAIN_ENTITIES
        self.TRAIN_CONFIDENCES = TRAIN_CONFIDENCES
        self.TRAIN_COSINE_SIM = TRAIN_COSINE_SIM
        self.TRAIN_CONTEXT = TRAIN_CONTEXT

        self.TEST_ENTITIES = TEST_ENTITIES
        self.TEST_CONFIDENCES = TEST_CONFIDENCES
        self.TEST_COSINE_SIM = TEST_COSINE_SIM
        self.TEST_CONTEXT = TEST_CONTEXT

        self.match_orig_feature = True
        self.print_query_scores = False

    def getFeatures(self, article_index, query_index, supporting_article_index, entities, confidences, cosine_sim, context):        
        features= []

        #Construct feature vector for this sampled entity
        original_confidence = confidences[article_index][query_index][0]
        confidence = confidences[article_index][query_index][supporting_article_index]
        
        # confidence_dif = [confidence[i] - original_confidence[i] for i in range(len(confidence))]
        # confidence_bool = [confidence[i] > original_confidence[i] for i in range(len(confidence))]

        # orig_confidence_thresh = [original_confidence[i] < .4 for i in range(len(confidence))]
        # confidence_thresh = [confidence[i] > .6 for i in range(len(confidence))]

        #One hot vector to show if entity matches orginal
        original_entity = entities[article_index][query_index][0]
        new_entity = entities[article_index][query_index][supporting_article_index]
        match_features = []
        for e_index in range(len(original_entity)):
            if original_entity[e_index] == '': # or original_entity[e_index] == 'unknown':
                match_features += [0, 0]
            elif original_entity[e_index].strip().lower() == new_entity[e_index].strip().lower():
                match_features += [1, 0]
            else:
                match_features += [0, 1]
        
        # Cosine sim array is shifted by one.
        # Index 0 should be 1 as orig is same as itself.
        tfidf = 1 if supporting_article_index == 0 else \
                cosine_sim[article_index]\
                [query_index][supporting_article_index - 1]


        features = original_confidence+ confidence + match_features + [tfidf]

        # if tfidf > .3:
        #     features += [1]
        # else:
        #     features += [0]

        # features += confidence_dif 
        # features += confidence_bool
        # features   += confidence_thresh
        # features   += orig_confidence_thresh

        for c in context[article_index][query_index][supporting_article_index]:
            features += c            
        
        return features

    def trainClassifier(self, train_identifiers, num_entities):
        classifier = MaxEnt(solver="lbfgs", verbose=1)
        X = []
        Y = []
        num_neg = 0
        max_neg = 1000 ## Set empiracally 
        for article_index in range(len(self.TRAIN_ENTITIES)):
            article = self.TRAIN_ENTITIES[article_index]
            for query_index in range(len(article)):
                query = article[query_index]
                for supporting_article_index in range(len(query)):
                    
                    features = self.getFeatures(article_index, query_index, supporting_article_index, \
                                            self.TRAIN_ENTITIES, self.TRAIN_CONFIDENCES,self.TRAIN_COSINE_SIM, self.TRAIN_CONTEXT)
                

                    if constants.mode == "EMA":
                        labels = self.getLabelsEMA(article_index, query_index, supporting_article_index, \
                            self.TRAIN_ENTITIES, train_identifiers)
                        none_desicion = 4
                    else:
                        labels = self.getLabelsShooter(article_index, query_index, supporting_article_index, \
                            self.TRAIN_ENTITIES, train_identifiers)
                        none_desicion = 5

                    for label in labels:
                        if label == none_desicion: 
                            if num_neg < max_neg:
                                num_neg+=1 
                                X.append(features)
                                Y.append(label)
                        else:
                            X.append(features)
                            Y.append(label)
        assert( len(X) == len(Y))
        classifier.fit(X,Y)
        print "Class dist", [sum([y == i for y in Y])for i in range(5)]
        print "Total labels", len(Y)   
        return classifier


    def getLabelsEMA(self, article_index, query_index, supporting_article_index, entities, identifier):
        #Extract out label for this article (ie. is label correct)
        labels = []
        gold_entities =     identifier[article_index]
        new_entities  =     entities[article_index][query_index][supporting_article_index]
        orig_entities =     entities[article_index][query_index][0]
        for ind in range(len(gold_entities)):
            ent = new_entities[ind].lower().strip()
            orig_ent = orig_entities[ind].lower().strip()
            gold = gold_entities[ind].lower().strip()
            

            match = evaluatePrediction(ent, gold)
            orig_match = evaluatePrediction(orig_ent, gold)

            if match == 1:
                labels.append(ind)
                # if not orig_match == 1:
                #     labels.append(ind)
            

        if set(labels) == set([0, 1, 2]):
            labels = [3]
        elif labels == []:
            labels = [4]
        
        assert (len(labels) > 0)
        return labels

    def getLabelsShooter(self, article_index, query_index, supporting_article_index, entities, identifier):
        #Extract out label for this article (ie. is label correct)
        labels = []
        gold_entities = identifier[article_index]
        new_entities      = entities[article_index][query_index][supporting_article_index]
        orig_entities     = entities[article_index][query_index][0]
        for ind in range(len(gold_entities)):
            ent = new_entities[ind].lower().strip()
            orig_ent = orig_entities[ind].lower().strip()
            gold = gold_entities[ind].lower().strip()
            if gold == "":
                continue
            if ent == "":
                continue
            
            #special handling for shooterName (entity_index = 0)
            if ind == 0:
                new_person = set(ent.split('|'))
                gold_person = set(gold.split('|'))
                if len(new_person.intersection(gold_person)) > 0:
                    if not ent == orig_ent:
                        labels.append(ind)
            else:
                if gold == ent:
                    if not ent == orig_ent:
                        labels.append(ind)
        if labels == [0, 1, 2, 3]:
            labels = [4]
        elif labels == []:
            labels = [5]
        
        assert (len(labels) > 0)
        return labels    

    def predictEntities(self, classifier, test_identifiers):
        if constants.mode == "Shooter":
            predictions = [0,0,0,0,0,0]
            take_all = 4
            num_ents = 4
        else:
            predictions = [0,0,0,0,0]
            take_all = 3
            num_ents = 3
        DECISIONS = copy.deepcopy(self.TEST_ENTITIES)
        i = 0
        for article_index in range(len(self.TEST_ENTITIES)):
            article = self.TEST_ENTITIES[article_index]
            for query_index in range(len(article)):
                query = article[query_index]
                for supporting_article_index in range(len(query)):
                    if supporting_article_index == 0:
                        DECISIONS[article_index][query_index]\
                            [supporting_article_index] = [1] * num_ents
                        continue
                    DECISIONS[article_index][query_index]\
                            [supporting_article_index] = [0] * num_ents 

                    features = self.getFeatures(article_index, query_index, supporting_article_index, self.TEST_ENTITIES, self.TEST_CONFIDENCES,\
                               self.TEST_COSINE_SIM, self.TEST_CONTEXT)
                    # print 'query[supporting_article_index]', query[supporting_article_index]
                    if query[supporting_article_index] == ['unknown', 'unknown', 'unknown'] or\
                     query[supporting_article_index] == ['', '', '', '']:
                        continue

                    prediction = classifier.predict(features)[0]
                    predictions[prediction] += 1
                    if prediction < take_all:
                        if query[supporting_article_index][prediction] == "unknown":
                            continue
                        DECISIONS[article_index][query_index]\
                            [supporting_article_index][prediction] = 1
                        # if prediction == 0 or prediction == 2:
                        #     print '------ in prediction -----'
                        #     print "Chose to replace"
                        #     print 'prediction is', prediction
                        #     print 'orig', query[0]
                        #     print 'supp', query[supporting_article_index]
                        #     print 'gold',test_identifiers[article_index]
                        #     print 'label', prediction
                        #     print
                        #     raw_input()
                    elif prediction == take_all:
                        if "unknown" in query[supporting_article_index]:
                            continue
                        DECISIONS[article_index][query_index]\
                            [supporting_article_index] = [1] * num_ents
        print "predictions", predictions

        return DECISIONS
    #Run both Max Confidence and Majority Aggregation Schemes given the decisions
    #Return the decided tag for each query
    def aggregateResults(self, DECISIONS, num_entities):
        majority = []
        max_conf = []    
        for article_index in range(len(self.TEST_ENTITIES)):
            for entity_index in range(num_entities):
                if entity_index == 0:
                    max_conf.append([])
                    majority.append([])
                article = self.TEST_ENTITIES[article_index]
                tag_occurances = {}
                max_confidence = -1
                max_confidence_tag = ''
                for query_index in range(len(article)):
                    query = article[query_index]   
                   
                    for supporting_article_index in range(len(query)):
                        supporting_article = query[supporting_article_index]
                        if DECISIONS[article_index][query_index][supporting_article_index][entity_index] == 0:
                            continue

                        confidence = self.TEST_CONFIDENCES[article_index][query_index]\
                                [supporting_article_index][entity_index]
                        entity = supporting_article[entity_index].strip().lower()

                        ##Update counts of majority
                        if entity not in tag_occurances:
                            tag_occurances[entity] = 1
                        else:
                            tag_occurances[entity] += 1

                        ##Update max_confidence
                        if confidence > max_confidence:
                            max_confidence = confidence
                            max_confidence_tag = entity
                    max_majority_count = -1
                majority_tag = ''
                for ent in tag_occurances:
                    if tag_occurances[ent] > max_majority_count:
                        max_majority_count = tag_occurances[ent]
                        majority_tag = ent
                max_conf[article_index].append(max_confidence_tag)
                majority[article_index].append(majority_tag)
        print 'len(majority)', len(majority)
        print 'len(DECISIONS)', len(DECISIONS)
        assert len(majority) == len(DECISIONS)
        assert len(max_conf) == len(majority)
        assert len(majority[0]) == num_entities
        return majority, max_conf



    def evaluateBaseline(self, predicted_identifiers, test_identifiers, num_entities, COUNT_ZERO):
        predicted_correct = [0.] * num_entities
        total_predicted   = [0.] * num_entities
        total_gold        = [0.] * num_entities
        
        print 'len(test_identifiers)', len(test_identifiers)
        print 'num_entities', num_entities
        for entity_index in range(num_entities):
            for article_index in range(len(predicted_identifiers)):
                predicted = predicted_identifiers[article_index][entity_index].strip().lower()
                gold = test_identifiers[article_index][entity_index].strip().lower()                    
                orig = self.TEST_ENTITIES[article_index][0][0][entity_index].strip().lower()
                match = evaluatePrediction(predicted, gold)

                if match == 'skip':
                    continue
                else:
                    total_gold[entity_index] += 1
                if match == "no_predict":
                    continue
                if match == 1:
                    predicted_correct[entity_index] += 1
                total_predicted[entity_index] += 1

        helper.printScores(predicted_correct, total_predicted,total_gold)


    def evaluateSansBaseline(self, predicted_identifiers, test_identifiers, num_entities, COUNT_ZERO):
        predicted_correct = [0.] * num_entities
        total_predicted   = [0.] * num_entities
        total_gold        = [0.] * num_entities
        
        print 'len(test_identifiers)', len(test_identifiers)
        print 'num_entities', num_entities
        for entity_index in range(num_entities):
            for article_index in range(len(predicted_identifiers)):
                predicted = self.TEST_ENTITIES[article_index][0][0][entity_index].strip().lower()
                gold = test_identifiers[article_index][entity_index].strip().lower()
                    
                match = evaluatePrediction(predicted, gold)

                if match == 'skip':
                    continue
                    
                else:
                    total_gold[entity_index] += 1
                if match == "no_predict":
                    continue
                if match == 1:
                    predicted_correct[entity_index] += 1
                total_predicted[entity_index] += 1

        helper.printScores(predicted_correct, total_predicted,total_gold)

   
    def trainAndEval(self, train_identifiers, test_identifiers, num_entities, COUNT_ZERO):
        classifier = self.trainClassifier(train_identifiers, num_entities)
        DECISIONS  = self.predictEntities(classifier, test_identifiers)
        print "#############################################################"
        print "Evaluation for Classifier baseline with SANS aggregation"
        self.evaluateSansBaseline(DECISIONS, test_identifiers, num_entities, COUNT_ZERO)
        majority, max_conf = self.aggregateResults(DECISIONS, num_entities)
        print "#############################################################"
        print "Evaluation for Classifier baseline with MAJORITY aggregation"
        self.evaluateBaseline(majority, test_identifiers, num_entities, COUNT_ZERO)
        print "#############################################################"
        print "Evaluation for Classifier baseline with MAX CONFIDENCE aggregation"
        self.evaluateBaseline(max_conf, test_identifiers, num_entities, COUNT_ZERO)
        