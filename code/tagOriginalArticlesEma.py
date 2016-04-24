from __future__ import unicode_literals
from nltk.tokenize import PunktWordTokenizer as WordTokenizer
import random
import pprint
import scipy.sparse
import time
import itertools
import sys
import pickle
import helper
import constants


tokenizer = WordTokenizer()
int2tags = constants.int2tags
tags2int = constants.tags2int
int2citationFeilds = ['Authors', 'Date', 'Title', 'Source']
generic = ["city", "centre", "county", "street", "road", "and", "in", "town", "village"]


def filterArticles(articles):
    relevant_articles = {}
    correct = [0] * (len(int2tags) -1 )
    gold_num = [0] * (len(int2tags)-1)
    filtered_correct = [0] * (len(int2tags) -1 )
    filtered_gold_num = [0] * (len(int2tags)-1)
    helper.load_constants()
    print "Num incidents", len(incidents)
    print "Num unfilitered articles", len(articles)
    for incident_id in incidents.keys():
        incident = incidents[incident_id]
        if not 'citations' in incident:
            continue
        for citation_ind, citation in enumerate(incident['citations']):
            saveFile = "../data/raw_data/"+ incident_id+"_"+str(citation_ind)+".raw"
            print "checking it for savefile", saveFile
            if not saveFile in articles:
                continue

            article = tokenizer.tokenize(articles[saveFile])
            ents = [incident[e.replace('-','_')] for e in int2tags[1:]]

            tags, cleanArticle = getTags(article, ents)

            ##Calculate scores for filitered and unflitered articles
            for i in range(1, len(int2tags) ):
                correct[i-1] += 1 if i in tags else 0
                gold_num[i-1]    += ents[i-1].strip().lower() not in ["unknown"]

            if len(set(tags)) > 2: ##This is the filtering
                for i in range(1, len(int2tags) ):
                    filtered_correct[i-1] += 1 if i in tags else 0
                    filtered_gold_num[i-1]    += ents[i-1].strip().lower() not in ["unknown"]
                #Store article in convenient format to writing to tagfile
                relavant_article = {}
                relavant_article['tokens'] = cleanArticle[:1000]
                relavant_article['tags']   = tags
                relavant_article['title']  = citation['Title']
                relavant_article['ents']   = [cleanDelimiters(e) for e in ents]
                relevant_articles[saveFile] = relavant_article
    pickle.dump(relevant_articles, open('EMA_filtered_articles.2.p', 'wb'))
                    
    oracle_scores = [(correct[i]*1./gold_num[i], int2tags[i+1]) if gold_num[i] > 0 else 0 for i in range(len(correct))]
    filtered_oracle_scores = [(filtered_correct[i]*1./filtered_gold_num[i], int2tags[i+1]) if filtered_gold_num[i] > 0 else 0 for i in range(len(correct))]
    print "num articles is", len(relevant_articles)
    print "oracle scores", oracle_scores
    print "filtered_oracle_scores", filtered_oracle_scores
    return relevant_articles

def cleanEnts(ent_tokens):
    ascii_tokens = asciiEnts(ent_tokens)
    result = [x.strip().lower() if (not x in generic and x.isalpha()) else "" for x in ascii_tokens]
    return result

def asciiEnts(ent_tokens):
    ascii_tokens = []
    for en in ent_tokens:
        try:
            ascii_tokens.append(en.encode("ascii", "ignore").lower())
        except Exception, e:
            ascii_tokens.append(en)
            pass
    return ascii_tokens

def cleanDelimiters(ent):
    result = ent.replace(',', '|')
    result = result.replace(';','|')
    return result.strip().lower()

def cleanToken(tok):
    tok = tok.lower().strip()
    if not tok[-1].isalpha():
        tok = tok[:-1]
    return tok

def getTags(article, ents):
    tags = []
    cleanArticle = []
    debug = False
    for i, token in enumerate(article):
        labels = []
        token = cleanToken(token)

        for j in range(len(ents)):
            ent = ents[j]
            if ent.lower() == "unknown":
                continue
            ent = cleanDelimiters(ent)
            ent_set = ent.split("|")
            for possible_ent in ent_set:
                possible_ents = tokenizer.tokenize(possible_ent)
                clean_possible_ents = cleanEnts(possible_ents)
                if token in clean_possible_ents and not token == '':
                    ind = asciiEnts(possible_ents).index(token)
                    context = [cleanToken(w) for w in article[ max(0,i-ind): min(len(article), i + len(possible_ents)- ind)]]
                    if debug:
                        print int2tags[j+1]
                        print "clean_ents", "tokens"
                        print clean_possible_ents
                        print "**"
                        print "context", context
                        print "cleanContext", cleanEnts(context)
                        print "-------"
                    if clean_possible_ents == cleanEnts(context):
                        labels.append(j+1)

        label = 0
        if len(labels) > 0:
            label = random.choice(labels)
        cleanArticle.append(token)
        tags.append(label)
    if debug:
        print "tag set", set(tags)
        inp = raw_input("Returning tags for articles. Press any to continue. \n")
        if inp.strip().lower() == "text":
            print "article", article
            print "---------"
            raw_input()
    return tags, cleanArticle

        



if __name__ == "__main__":

    random.seed(1)
    train = open('../data/tagged_data/EMA/train.tag', 'wb')
    dev = open('../data/tagged_data/EMA/dev.tag', 'wb')
    test = open('../data/tagged_data/EMA/test.tag', 'wb')

    
    incidents = pickle.load(open('EMA_dump.p', 'rb'))
    downloaded_articles = pickle.load(open('EMA_downloaded_articles_dump.p.server', 'rb'))

    train_fraction = .60
    dev_fraction   = .10
    test_fraction  = .30

    refilter = True
    if refilter:
        relevant_articles = filterArticles(downloaded_articles)
        raw_input("Done filtering. Press anything to continue")
    else:
        relevant_articles = pickle.load(open('EMA_filtered_articles.2.p', 'rb'))
        print "Using relevant articles from EMA_filtered_articles.2.p"

    num_articles = len(relevant_articles)
    for saveFile in relevant_articles:
        relevant_article = relevant_articles[saveFile]
        tokens, tags, title, ents = relevant_article['tokens'], relevant_article['tags'], relevant_article['title'],\
         relevant_article['ents']

        tagged_body = ""
        for tok, tag in zip(tokens, tags):
            try:
                tagged_tok   = tok + "_" + int2tags[tag]
                assert not tok == " "
                assert len(tagged_tok.split("_")) >= 2
                tagged_body += tagged_tok + " "
            except Exception, e:
                pass
        try:
            identifier = ','.join(ents)+', ' + title
        except Exception, e:
            print 'ents', ents
            raw_input()

        

        #Assign to train/dev/test
        partition = random.random()
        if partition < train_fraction:
            f = train
        elif partition < (train_fraction + dev_fraction):
            f = dev
        else:
            f = test

        #Ascii encode ents 
        try:
            ents_str = ','.join(ents) + ', '
            f.write(ents_str)
        except Exception, e:
            ents_str = "".join(asciiEnts(ents_str))
            f.write(ents_str)
            pass
        
        title = " ".join(cleanEnts(tokenizer.tokenize( title)))
        f.write( title + '\n')
        f.write(tagged_body + '\n')
        f.flush()

    train.close()
    dev.close()
    test.close()


