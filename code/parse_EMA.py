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

tokenizer = WordTokenizer()
int2tags = \
['TAG',\
'Affected_Food_Product',\
'Produced_Location',\
'Distributed_Location']
tags2int = \
{'TAG':0,\
'Affected_Food_Product':1, \
'Produced_Location':2, \
'Distributed_Location':3 }
int2citationFeilds = ['Authors', 'Date', 'Title', 'Source']
generic = ["city", "centre", "county", "street", "road", "and", "in", "town", "village"]


def filterArticles(articles):
    relevant_articles = {}
    count = 0
    correct = [0] * len(int2tags)
    gold_num = [0] * len(int2tags)
    helper.load_constants()
    print "Num incidents", len(incidents)
    print "Num unfilitered articles", len(articles)
    for incident_id in incidents.keys():
        incident = incidents[incident_id]
        if not 'citations' in incident:
            continue
        for citation_ind, citation in enumerate(incident['citations']):
            saveFile = "../data/raw_data/"+ incident_id+"_"+str(citation_ind)+".raw"
            if not saveFile in articles:
                continue
            count +=1
            article = articles[saveFile]
            for ent in int2tags:
                if not ent in incident:
                    continue
                gold = incident[ent]

                if ent in ['Consumer_Brand', 'Perpetrator', 'Adulterated_Food_Product', 'Affected_Food_Product']:
                    gold_list = gold.split(';')
                elif ent in ["Produced_Location", "Distributed_Location"]:
                    country = gold.split(',')
                    gold_list = gold.split(',')
                else:
                    gold_list = [gold]

                for g in gold_list:
                    g = g.lower().strip()
                    if g in ['', 'none', 'unknown', "0"]:
                        continue
                    clean_g = g.encode("ascii", "ignore")
                    clean_article = ""
                    for c in article:
                        try:
                            clean_article += c.encode("ascii", "ignore")
                        except Exception, e:
                            clean_article += ""

                    if clean_g in clean_article.lower():
                        if not saveFile in relevant_articles:
                            relevant_articles[saveFile] = clean_article
                        correct[tags2int[ent]] += 1
                gold_num[tags2int[ent]] += 1

    pickle.dump(downloaded_articles, open('EMA_filtered_articles.p', 'wb'))
                    
    oracle_scores = [(correct[i]*1./gold_num[i], int2tags[i]) if gold_num[i] > 100 else 0 for i in range(len(int2tags))]
    print "num articles is", len(relevant_articles)
    return relevant_articles, oracle_scores

def cleanEnts(ent_tokens):
    return [x.strip().lower() if (not x in generic and x.isalpha()) else "" for x in ent_tokens]

def getTags(article, ents):
    tags = []
    for i, token in enumerate(article):
        labels = []
        token = token.lower().strip()
        for j, ent in enumerate(ents):
            ent = ent.strip().lower()
            ent_tokens = tokenizer.tokenize(ent)
            if "|" in ent:
                ent_set = ent.split("|")
                for possible_ent in ent_set:
                    possible_ents = tokenizer.tokenize(possible_ent)
                    clean_possible_ents = cleanEnts(possible_ents)
                    if token in clean_possible_ents:
                        ind = possible_ents.index(token)
                        context = article[ max(0,i-ind): max(len(article), i + len(possible_ents)- ind)]
                        if clean_possible_ents in cleanEnts(context):
                            labels.append(j+1)
                            break
            elif len(ent_tokens) > 1:
                cleaned_ent_tokens = cleanEnts(ent_tokens)
                if token in cleaned_ent_tokens:
                    ind = ent_tokens.index(token)
                    context = article[ max(0,i-ind): max(len(article), i + len(ent_tokens)- ind)]
                    if cleaned_ent_tokens in cleanEnts(context):
                        labels.append(j+1)
            else:
                if ent.lower().strip() in token.lower().strip():
                    labels.append(j+1)
        label = 0
        if len(labels) == 1:
            label = labels[0]
        tags.append(label)
    return tags

        



if __name__ == "__main__":

    train = open('../data/tagged_data/EMA/train.tag', 'w')
    dev = open('../data/tagged_data/EMA/dev.tag', 'w')
    test = open('../data/tagged_data/EMA/test.tag', 'w')
    
    incidents = pickle.load(open('EMA_dump.p', 'rb'))
    downloaded_articles = pickle.load(open('EMA_downloaded_articles_dump.p', 'rb'))

    train_cut = .60
    dev_cut   = .20
    test_cut  = .20

    refilter = False
    if refilter:
        relevant_articles, unfilitered_scores = filterArticles(downloaded_articles)
        pprint.pprint(unfilitered_scores)
    else:
        relevant_articles = pickle.load(open('EMA_filtered_articles.p', 'rb'))

    ratios = {}
    correct = [0] * (len(int2tags)-1)
    for ind, incident_id in enumerate(incidents.keys()):
        print ind,'/',len(incidents.keys())
        incident = incidents[incident_id]
        if not 'citations' in incident:
            continue
        for citation_ind, citation in enumerate(incident['citations']):
            title = incident['citations'][citation_ind]['Title']
            saveFile = "../data/raw_data/"+ incident_id+"_"+str(citation_ind)+".raw"
            if not saveFile in relevant_articles:
                continue
            article = relevant_articles[saveFile]
            tokens = tokenizer.tokenize(article)
            ents = []
            for ent in int2tags[1:]:
                if ent in incident:
                    gold = incident[ent]
                    if ent in ['Affected_Food_Product']:
                        gold_list = gold.split(';')
                    elif ent in ["Produced_Location", "Distributed_Location"]:
                        gold_list = []
                        locations =  gold.split(',')
                        for loc in locations:
                            gold_list += loc.split(';')
                    else:
                        gold_list = [gold]
                    ents.append("|".join(gold_list))
                else:
                    ents.append('')
            tags = getTags(tokens, ents)

            for ent_ind in range(1,len(int2tags)):
                if ent_ind in tags:
                    correct[ent_ind - 1] += 1

            tagged_body = ""
            for token, tag in zip(tokens, tags):
                try:
                    tagged_body += token + "_" + int2tags[tag] + " "
                except Exception, e:
                    tagged_body += ""
            out_ident = ','.join(ents) + ", " + title

            rand = random.random()
            f = ''
            if rand < train_cut:
                f = train
            elif rand < dev_cut:
                f = dev
            else:
                f = test

            cleaned_identifier = out_ident
            cleaned_body = tagged_body
            try:
                f.write(cleaned_identifier + '\n')
            except Exception, e:
                new_ident = ""
                for c in cleaned_identifier:
                    try:
                        new_ident += c.encode("ascii", "ignore")
                    except Exception, e:
                        new_ident += ""
                f.write(new_ident + '\n')
            
            try:
                f.write(cleaned_body + '\n')
            except Exception, e:
                new_body = ""
                for c in cleaned_identifier:
                    try:
                        new_body += c.encode("ascii", "ignore")
                    except Exception, e:
                        new_body += ""
                f.write(new_body + '\n')
                
            f.flush()

    train.close()
    dev.close()
    test.close()
    ratios =[(correct[i] * 1. / len(relevant_articles), int2tags[i+1]) for i in range(len(correct))]
    pprint.pprint(ratios)

