from sklearn.linear_model import LogisticRegression as MaxEnt
import copy

class Classifier(object):

    def __init__(self, TRAIN_ENTITIES, TRAIN_CONFIDENCES, TRAIN_COSINE_SIM,\
                 TEST_ENTITIES, TEST_CONFIDENCES, TEST_COSINE_SIM):
        self.TRAIN_ENTITIES = TRAIN_ENTITIES
        self.TRAIN_CONFIDENCES = TRAIN_CONFIDENCES
        self.TRAIN_COSINE_SIM = TRAIN_COSINE_SIM

        self.TEST_ENTITIES = TEST_ENTITIES
        self.TEST_CONFIDENCES = TEST_CONFIDENCES
        self.TEST_COSINE_SIM = TEST_COSINE_SIM


    def trainClassifiers(self, train_identifiers, num_entities):
        classifiers = [] ##List of classifiers

        for entity_index in range(num_entities):
            classifiers.append(MaxEnt(multi_class='ovr', solver='lbfgs'))
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

                        features = [original_confidence, confidence] + entity_match + [tfidf]


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

                        features = [original_confidence, confidence] + entity_match + [tfidf]


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
            predicted_correct = 0.
            total_predicted   = 0.
            total_gold        = 0.
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
                        predicted_correct += (1 if len(correct)>0 else 0)
                        total_predicted += 1
                        total_gold += 1 
                    else:
                        total_predicted += 1
                        if predicted == gold:
                            predicted_correct += 1
                        total_gold += 1

            print "Entity", entity_index, ":",
            if total_predicted == 0 :
                continue

            if predicted_correct == 0 :
                continue

            precision = predicted_correct / total_predicted
            recall = predicted_correct / total_gold
            f1 = (2*precision*recall)/(precision+recall)
            print "PRECISION", precision, "RECALL", recall, "F1", f1
            print "Total predicted", total_predicted


    def trainAndEval(self, train_identifiers, test_identifiers, num_entities, COUNT_ZERO):
        classifiers = self.trainClassifiers(train_identifiers, num_entities)
        DECISIONS  = self.predictEntities(classifiers, num_entities)
        majority, max_conf = self.aggregateResults(DECISIONS, num_entities)

        print "Evaluation for Classifier baseline with MAJORITY aggregation"
        self.evaluateBaseline(majority, test_identifiers, num_entities, COUNT_ZERO)


        print "Evaluation for Classifier baseline with MAX CONFIDENCE aggregation"
        self.evaluateBaseline(max_conf, test_identifiers, num_entities, COUNT_ZERO)
