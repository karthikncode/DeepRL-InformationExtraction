import json
import pickle
import constants

MODE= constants.mode
# MODE='Shooter'

def load_constants():
    global male_first_names,female_first_names,last_names,cities,other_features,number_as_words,word_ordinals, other_features_names, \
        adulterants, foods

    cities = pickle.load(open('../data/constants/cities.p','rb'))

    if MODE == 'Shooter':
        with open('../data/constants/male_first_names.json','rb') as outfile:
            male_first_names = set(json.load(outfile))
        with open('../data/constants/female_first_names.json','rb') as outfile:
            female_first_names = set(json.load(outfile))
        with open('../data/constants/last_names.json','rb') as outfile:
            last_names = set(json.load(outfile))
        with open('../data/constants/train_names.json','rb') as outfile:
            train_names = set(json.load(outfile))
        with open('../data/constants/number_as_words.json','rb') as outfile:
            number_as_words = set(json.load(outfile))
        with open('../data/constants/word_ordinals.json','rb') as outfile:
            word_ordinals = set(json.load(outfile))
        other_features = [is_capital,is_digit,is_male_filsrst_name,is_female_first_name,is_last_name,is_full_city,is_partial_city,contains_digit,is_short_word,is_long_word,is_number_word,is_ordinal_word,is_ordinal_num, is_adulterant]
        other_feature_names = ['is_capital','is_digit','is_male_first_name','is_female_first_name','is_last_name','is_full_city','is_partial_city','contains_digit','is_short_word','is_long_word','is_number_word','is_ordinal_word','is_ordinal_num', 'is_adulterant']
    #other_features = []
    elif MODE == 'EMA':
        with open('../data/constants/adulterants.p','rb') as outfile:
            adulterants = set(pickle.load(outfile))
        with open('../data/constants/foods.p','rb') as outfile:
            foods = pickle.load(outfile)
        with open('../data/constants/number_as_words.json','rb') as outfile:
            number_as_words = set(json.load(outfile))
        with open('../data/constants/word_ordinals.json','rb') as outfile:
            word_ordinals = set(json.load(outfile))
        other_features = [is_capital,is_full_city,is_partial_city,is_short_word,is_long_word,is_number_word,is_ordinal_word,is_ordinal_num, is_adulterant, is_food]
        other_feature_names = ['is_capital','is_full_city','is_partial_city','is_short_word','is_long_word','is_number_word','is_ordinal_word','is_ordinal_num',' is_adulterant', 'is_food']


    #Ideas: is_country, is_partial_country, is food, is_partial_food, is_aduletrant, is_partial_adulteran, remove_number_shit, 
    # is_region, is_partial_region, 

    #TODO: check is_food()
    
    #IMP: adding train names to last names here. Comment below if you don't want this
    # last_names.update(train_names)




# other features, return true or false
def is_capital(word):
    return word[0].isupper()

def is_adulterant(word):
    return word.lower() in adulterants

def is_food(word):
    return word.lower() in foods

def is_digit(word):
    return word.isdigit()

def is_male_first_name(word):
	return word.lower() in male_first_names

def is_female_first_name(word):
    return word.lower() in female_first_names

def is_last_name(word):
    return word.lower() in last_names

def is_full_city(word):
    return word in cities and "" in cities[word]

def is_partial_city(word):
    return word in cities and "" not in cities[word]

def contains_digit(word):
    return any(char.isdigit() for char in word)

def is_short_word(word):
    return len(word) < 4

def is_long_word(word):
    return len(word) > 8

def is_number_word(word):
    return word in number_as_words

def is_ordinal_word(word):
    return word in word_ordinals

def is_ordinal_num(word):
    return contains_digit(word) and (word.endswith('th') or word.endswith('nd') or word.endswith('st'))

def captilized(word):
    return word.istitle()

def getOtherFeatures(word):
    features = {}
    for index, feature_func in other_features:
        name = other_features_names[index]
        features[name] = feature_func(word)

    return features

def printScores(correct, guessed, gold_c):
    num_tags = len(constants.int2tags) - 1
    print "tag_type (correct, guessed, gold) (percision, recall, f1)"
    for k in range(num_tags): 
        evalText = getPrecRecallF1String(correct[k], guessed[k], gold_c[k], constants.int2tags[k+1])
        print evalText

def getPrecRecallF1String(correct, guessed, gold, tag):
    percision = 1.*correct/guessed if guessed > 0 else 0 
    recall = 1.*correct/gold if guessed > 0 else 0
    f1 = (2.*percision*recall)/(percision+recall) \
        if (percision+recall) > 0 else 0

    evalText =  tag + ' ( ' + str(correct) + ", " + str(guessed) + ", " + str(gold) + ")"
    evalText += '( ' + str(percision) + ", " + str(recall) + ", " + str(f1) + ")"
    return evalText
