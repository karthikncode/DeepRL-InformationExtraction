from sklearn.linear_model import LogisticRegression as MaxEnt
import copy
import random
import collections
from itertools import izip



class Classifier(object):

    def __init__(self, TRAIN_ENTITIES, TRAIN_CONFIDENCES, TRAIN_COSINE_SIM,\
                 TEST_ENTITIES, TEST_CONFIDENCES, TEST_COSINE_SIM):
        self.TRAIN_ENTITIES = TRAIN_ENTITIES
        self.TRAIN_CONFIDENCES = TRAIN_CONFIDENCES
        self.TRAIN_COSINE_SIM = TRAIN_COSINE_SIM

        self.TEST_ENTITIES = TEST_ENTITIES
        self.TEST_CONFIDENCES = TEST_CONFIDENCES
        self.TEST_COSINE_SIM = TEST_COSINE_SIM

        self.match_orig_feature = True
        self.print_query_scores = False


    def trainClassifiers(self, train_identifiers, num_entities):
        classifiers = [] ##List of classifiers

        for entity_index in range(num_entities):
            classifiers.append(MaxEnt())
            X = []
            Y = []
            for article_index in range(len(self.TRAIN_ENTITIES)):
                article = self.TRAIN_ENTITIES[article_index]
                for query_index in range(len(article)):
                    query = article[query_index]
                    for supporting_article_index in range(len(query)):
                        supporting_article = query[supporting_article_index]

                        #Construct feature vector for this sampled entity
                        original_confidence = self.TRAIN_CONFIDENCES[article_index][query_index]\
                                [0][entity_index]
                        confidence = self.TRAIN_CONFIDENCES[article_index][query_index]\
                                [supporting_article_index][entity_index]
                        original_entity = query[0][entity_index].strip().lower()
                        entity = supporting_article[entity_index].strip().lower()
                        if entity == '':
                            continue
                        entity_match = [1, 0] if original_entity == entity else [0, 1]

                        # Cosine sim array is shifted by one.
                        # Index 0 should be 1 as orig is same as itself.
                        tfidf = 1 if supporting_article_index == 0 else \
                                self.TRAIN_COSINE_SIM[article_index]\
                                [query_index][supporting_article_index - 1]

                        if self.match_orig_feature:
                            features = [original_confidence, confidence] + entity_match + [tfidf]
                        else:
                            features = [original_confidence, confidence] + [tfidf]

                        #Extract out label for this article (ie. is label correct)
                        gold_entity = train_identifiers[article_index]\
                                      [entity_index].strip().lower()
                        if gold_entity == '':
                            continue

                        #special handling for shooterName (entity_index = 0)
                        if entity_index == 0:
                            entity = set(entity.split('|'))
                            gold_entity = set(gold_entity.split('|'))
                            if len(entity.intersection(gold_entity)) > 0:
                                label = 1
                            else:
                                label = 0
                        else:
                            label = int(gold_entity == entity)

                        ## Only if gold entity and the entity aren't empty,
                        ## we add this as a training example
                        X.append(features)
                        Y.append(label)
            assert( len(X) == len(Y))

            classifiers[entity_index].fit(X,Y)
            print "CLASSIFIER INDEX", entity_index
            print "Ratio of Labels being ones is ", sum(Y)*1./len(Y)
            
        return classifiers

    def predictEntities(self, classifiers, num_entities):
        assert(len(classifiers) == num_entities)
        DECISIONS = copy.deepcopy(self.TEST_ENTITIES)
        for entity_index in range(num_entities):
            for article_index in range(len(self.TEST_ENTITIES)):
                article = self.TEST_ENTITIES[article_index]
                for query_index in range(len(article)):
                    query = article[query_index]
                    for supporting_article_index in range(len(query)):
                        supporting_article = query[supporting_article_index]

                        #Construct feature vector for this sampled entity
                        original_confidence = self.TEST_CONFIDENCES[article_index][query_index]\
                                [0][entity_index]
                        confidence = self.TEST_CONFIDENCES[article_index][query_index]\
                                [supporting_article_index][entity_index]
                        original_entity = query[0][entity_index].strip().lower()
                        entity = supporting_article[entity_index].strip().lower()
                        entity_match = [1, 0] if original_entity == entity else [0, 1]

                        # Cosine sim array is shifted by one.
                        # Index 0 should be 1 as orig is same as itself.
                        tfidf = 1 if supporting_article_index == 0 else \
                                self.TEST_COSINE_SIM[article_index]\
                                [query_index][supporting_article_index - 1]

                        if self.match_orig_feature:
                            features = [original_confidence, confidence] + entity_match + [tfidf]
                        else:
                            features = [original_confidence, confidence] +  [tfidf]

                        if entity == '':
                            DECISIONS[article_index][query_index]\
                                [supporting_article_index][entity_index] = 0
                        else:
                            prediction = classifiers[entity_index].predict(features)[0]
                            DECISIONS[article_index][query_index]\
                                [supporting_article_index][entity_index] = prediction
        return DECISIONS

    #Run both Max Confidence and Majority Aggregation Schemes given the decisions
    #Return the decided tag for each query
    def aggregateResults(self, DECISIONS, num_entities):
        majority = []
        max_conf = []
        for article_index in range(len(self.TEST_ENTITIES)):
            max_conf.append([])
            majority.append([])
            article = self.TEST_ENTITIES[article_index]
            for query_index in range(len(article)):
                max_conf[article_index].append([])
                majority[article_index].append([])
                query = article[query_index]
                for entity_index in range(num_entities):
                    max_confidence = -1
                    max_confidence_tag = ''
                    tag_occurances = {}
                    for supporting_article_index in range(len(query)):
                        supporting_article = query[supporting_article_index]
                        if DECISIONS[article_index][query_index][supporting_article_index]\
                           [entity_index] == 0:
                            continue


                        confidence = self.TEST_CONFIDENCES[article_index][query_index]\
                                [supporting_article_index][entity_index]
                        entity = supporting_article[entity_index].strip().lower()
                        assert(not entity == '')

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
                    max_conf[article_index][query_index].append(max_confidence_tag)
                    majority[article_index][query_index].append(majority_tag)

        return majority, max_conf



    def evaluateBaseline(self, predicted_identifiers, test_identifiers, num_entities, COUNT_ZERO):
        for entity_index in range(num_entities):
            num_queries = 5
            predicted_correct = [0.] * num_queries
            total_predicted   = [0.] * num_queries
            total_gold        = [0.] * num_queries

            for article_index in range(len(predicted_identifiers)):
                ## TODO: Add classifier for selecting query index?
                for query_index in range(len(predicted_identifiers[article_index])):        
                    predicted = predicted_identifiers[article_index][query_index][entity_index].strip().lower()
                    gold = test_identifiers[article_index][entity_index].strip().lower()
                    if gold == '' or (not COUNT_ZERO and gold == 'zero'):
                        continue


                    #special handling for shooterName (lenient eval)
                    if entity_index == 0:
                        predicted = set(predicted.split('|'))
                        gold = set(gold.split('|'))
                        correct = gold.intersection(predicted)
                        predicted_correct[query_index] += (1 if len(correct)>0 else 0)
                        total_predicted[query_index] += 1
                        total_gold[query_index] += 1 
                    else:
                        total_predicted[query_index] += 1
                        if predicted == gold:
                            predicted_correct[query_index] += 1
                        total_gold[query_index] += 1


            print "Entity", entity_index, ":",
            if sum(total_predicted) == 0 :
                continue

            if sum(predicted_correct) == 0 :
                continue

            if  self.print_query_scores:
                print "BEGINNING WITH PER QUERY SCORES"

                for query_index in range(num_queries):
                    print "*********************************************"
                    print
                    print "QUERY INDEX:", query_index
                    self.displayScore(predicted_correct[query_index], total_predicted[query_index],\
                                      total_gold[query_index])
                    print
                    print "*********************************************"
                print "NOW SHOWING SCORES AGGREGATED OVER ALL QUERRIES"
            self.displayScore(sum(predicted_correct), sum(total_predicted),sum(total_gold))

    def displayScore(self, predicted_correct, total_predicted, total_gold):
        precision = predicted_correct / total_predicted
        recall = predicted_correct / total_gold
        f1 = (2*precision*recall)/(precision+recall)
        print "PRECISION", precision, "RECALL", recall, "F1", f1
        print "Total predicted", total_predicted

    def runExploratoryTests(self, DECISIONS, train_identifiers, test_identifiers):
        print "Exploring how many times gold entity is not in original document"
        count = collections.defaultdict(lambda:0.)
        total_count = collections.defaultdict(lambda:0.)
        for article_index in range(len(self.TRAIN_ENTITIES)):
            article = self.TRAIN_ENTITIES[article_index]
            for entity_index in range(4):
                for query_index in range(len(article)):
                    query = article[query_index]
                    if query_index > 0: #not shooter
                        orig_entity = query[0][entity_index].strip().lower()
                        gold = train_identifiers[article_index][entity_index].strip().lower()
                        for supp_index in range(len(query)):
                            entity = query[supp_index][entity_index].strip().lower()
                            if entity == gold:
                                count[entity_index] += 1
                            total_count[entity_index] +=1
                    else:
                        orig_entity = set(query[0][entity_index].strip().lower().split('|'))
                        gold = set(train_identifiers[article_index][entity_index].strip().lower().split('|'))
                        for supp_index in range(len(query)):
                            entity = set(query[supp_index][entity_index].strip().lower().split('|'))
                            if len(entity.intersection(gold)) > len(orig_entity.intersection(gold)):
                                count[entity_index] += 1
                            total_count[entity_index] +=1

        print "COUNT ", count
        print "TOTAL ", total_count
        print "Ratio" , [a/b for a,b in izip(count.values(),total_count.values())]

        print "Exploring if classifier ever chooses not first entity"
        print "Program will halt with assert if classifier chooses entity not in org document"
        ones = [0] * 4
        ones_not_orig = [0] * 4
        counts = [0] * 4
        for entity_index in range(4):
            for article_index in range(len(self.TEST_ENTITIES)):
                article = self.TEST_ENTITIES[article_index]
                for query_index in range(len(article)):
                    query = article[query_index]
                    orig_entity = query[0][entity_index].strip().lower()
                    gold = test_identifiers[article_index][entity_index].strip().lower()
                    for supp_index in range(len(query)):
                        decision = DECISIONS[article_index][query_index][supp_index][entity_index]
                        ones[entity_index] += decision
                        counts[entity_index] += 1
                        if decision == 1:
                            entity = query[supp_index][entity_index].strip().lower()
                            if entity != orig_entity:
                                ones_not_orig[entity_index] += 1

        print "Ratio ones in prediction", [ ones[x]*1. / counts[x] for x in range(4)]
        print "Ratio one not matching original entity in prediction", [ ones_not_orig[x]*1. / counts[x] for x in range(4)]
        print "It does not"


    def trainAndEval(self, train_identifiers, test_identifiers, num_entities, COUNT_ZERO):
        classifiers = self.trainClassifiers(train_identifiers, num_entities)
        DECISIONS  = self.predictEntities(classifiers, num_entities)

        debug = False
        if debug:
            self.runExploratoryTests(DECISIONS, train_identifiers, test_identifiers)
            return

        majority, max_conf = self.aggregateResults(DECISIONS, num_entities)
        print "#############################################################"
        print "#############################################################"
        print "Evaluation for Classifier baseline with MAJORITY aggregation"
        print
        self.evaluateBaseline(majority, test_identifiers, num_entities, COUNT_ZERO)

        print
        print "#############################################################"
        print "#############################################################"

        print "#############################################################"
        print "#############################################################"
        print "Evaluation for Classifier baseline with MAX CONFIDENCE aggregation"
        print
        self.evaluateBaseline(max_conf, test_identifiers, num_entities, COUNT_ZERO)
        print
        print "#############################################################"
        print "#############################################################"
