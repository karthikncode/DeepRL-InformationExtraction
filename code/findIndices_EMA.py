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

                if ent in ['Affected_Food_Product']:
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
                    clean_tokens = []
                    for c in tokenizer.tokenize(article):
                        try:
                            clean_tokens.append(c.encode("ascii", "ignore"))
                        except Exception, e:
                            pass
                    clean_article = " ".join(clean_tokens)
                    if clean_g in clean_article.lower():
                        if not saveFile in relevant_articles:
                            relevant_articles[saveFile] = clean_article.lower()
                        if False:
                            print clean_g
                            ind = clean_article.lower().index(clean_g)
                            print clean_article[max(0, ind - 100): min(len(clean_article), ind + 100)]
                        correct[tags2int[ent]] += 1
                gold_num[tags2int[ent]] += 1

    pickle.dump(downloaded_articles, open('EMA_filtered_articles.p', 'wb'))
                    
    oracle_scores = [(correct[i]*1./gold_num[i], int2tags[i]) if gold_num[i] > 100 else 0 for i in range(len(int2tags))]
    print "num articles is", len(relevant_articles)
    return relevant_articles, oracle_scores

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
            # print "that should not be possible"
            # print en
            ascii_tokens.append(en)
            pass
    return ascii_tokens
def getTags(article, ents):
    tags = []
    for i, token in enumerate(article):
        labels = []
        token = token.lower().strip()
        for j in range(len(ents)):
            ent = ents[j]

            if "|" in ent:
                ent_set = ent.split("|")
                for possible_ent in ent_set:
                    possible_ents = tokenizer.tokenize(possible_ent)
                    clean_possible_ents = cleanEnts(possible_ents)
                    if token in clean_possible_ents:
                        ind = asciiEnts(possible_ents).index(token)
                        context = article[ max(0,i-ind): min(len(article), i + len(possible_ents)- ind)]
                        if False:
                            print "clean_ents", "tokens"
                            print clean_possible_ents
                            print "**"
                            print cleanEnts(context)
                            print "-------"
                        if clean_possible_ents == cleanEnts(context):
                            labels.append(j+1)
                            break
            else:
                ent_tokens = tokenizer.tokenize(ent)
                if len(ent_tokens) > 1:
                    cleaned_ent_tokens = cleanEnts(ent_tokens)
                    if token in cleaned_ent_tokens:
                        ind = asciiEnts(ent_tokens).index(token)
                        context = article[ max(0,i-ind): min(len(article), i + len(ent_tokens)- ind)]
                        if False:
                            print "clean_ents_tokens", "tokens"
                            print cleaned_ent_tokens
                            print "**"
                            print cleanEnts(context)
                            print "-------"
                        if cleaned_ent_tokens == cleanEnts(context):
                            labels.append(j+1)
                else:
                    try:
                        if cleanEnts([ent]) == cleanEnts([token]):
                            labels.append(j+1)
                    except Exception, e:
                        pass

        label = 0
        if len(labels) > 0:
            label = random.choice(labels)
        tags.append(label)
    return tags

        



if __name__ == "__main__":

    train = open('../data/tagged_data/EMA2/train.tag', 'r')
    dev = open('../data/tagged_data/EMA2/dev.tag', 'r')
    test = open('../data/tagged_data/EMA2/test.tag', 'r')
    train_tmp = open('../data/tagged_data/EMA/train.tag.tmp', 'w')
    # dev_tmp = open('../data/tagged_data/EMA/dev.tag.tmp', 'r')
    # test_tmp = open('../data/tagged_data/EMA/test.tag.tmp', 'r')

    idents_split = {}
    
    incidents = pickle.load(open('EMA_dump.p', 'rb'))
    downloaded_articles = pickle.load(open('EMA_downloaded_articles_dump.p.server', 'rb'))

    train_cut = .60
    dev_cut   = .20
    test_cut  = .20
    train_index = 0
    dev_index   = 0
    test_index  = 0
    
    lastRead = {'train':False, 'dev':False, 'test':False}
    splitSets = {'train':set(), 'dev':set(), 'test':set()}
    splitCounter = {'train':0, 'dev':0, 'test':0}
    outCounter = {'train':0, 'dev':0, 'test':0}

    # count_fuckthisshit = 0
    # count_all = 0
    for f, i in [(train, 'train'), (dev, 'dev'), (test, 'test' )]:
        count_fuckthisshit = 0
        count_all = 0
        f_ident = f.readline().strip()
        f_body  = f.readline().strip()
        while not f_ident == "":            
            starter = idents_split[f_ident] if f_ident in idents_split else 0
            if not starter == 0:
                print "**********"
                print idents_split[f_ident] if f_ident in idents_split else 0
            if f_ident in idents_split:
                count_fuckthisshit +=1
                idents_split[f_ident].append((i, splitCounter[i]))
            else:
                idents_split[f_ident] = [(i, splitCounter[i])]
            if not starter == 0:
                print idents_split[f_ident]
                print "--------------"
                raw_input()
            count_all +=1
            splitSets[i].add(f_ident)
            splitCounter[i] += 1
            f_ident = f.readline().strip()
            f_body  = f.readline().strip()
        # print "For my main set", i
        # print "Umm, so our sets had x many duplicates, which is fucked, and x is", count_fuckthisshit , "/", count_all


    print "num idents", len(idents_split)
    for ident in idents_split.keys():
        occ = idents_split[ident]
        for part, index in occ:
            outCounter[part] += 1
    print "outCounter", outCounter

    print splitSets['train'].intersection(splitSets['dev'])
    print splitSets['train'].intersection(splitSets['test'])
    print splitSets['dev'].intersection(splitSets['test'])

    print "lens"
    print 'train', len(splitSets['train'])
    print 'dev', len(splitSets['dev'])
    print 'test',len(splitSets['test'])
    pprint.pprint(splitCounter)
    raw_input()





    refilter = False
    if refilter:
        relevant_articles, unfilitered_scores = filterArticles(downloaded_articles)
        pprint.pprint(unfilitered_scores)
    else:
        relevant_articles = pickle.load(open('EMA_filtered_articles.p.server', 'rb'))

    ratios = {}
    correct = [0] * (len(int2tags)-1)
    count = 0
    for ind, incident_id in enumerate(incidents.keys()[20:]):
        print ind,'/',len(incidents.keys())
        incident = incidents[incident_id]
        if not 'citations' in incident:
            continue
        ents = []
        for ent in int2tags[1:]:
                if ent in incident:
                    gold = incident[ent].encode('ascii', 'ignore')
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
        pprint.pprint(ents)
        for citation_ind, citation in enumerate(incident['citations']):
            title = incident['citations'][citation_ind]['Title']
            saveFile = "../data/raw_data/"+ incident_id+"_"+str(citation_ind)+".raw"
            if not saveFile in relevant_articles:
                continue
            article = relevant_articles[saveFile]
            #raw_input()
            tokens = tokenizer.tokenize(article)[:1000]
            
            tags = getTags(tokens, ents)
            correct_pass = [0] * (len(int2tags)-1)
            for ent_ind in range(1,len(int2tags)):
                if ent_ind in tags:
                    correct_pass[ent_ind - 1] += 1

            if sum(correct_pass) < 1 :
                continue

            for c_i, c in enumerate(correct_pass):
                correct[c_i] += c
            count += 1
            pprint.pprint(correct_pass)
            tagged_body = ""
            for token, tag in zip(tokens, tags):
                try:
                    tagged_body += token + "_" + int2tags[tag] + " "
                except Exception, e:
                    tagged_body += ""
            out_ident = ','.join(ents) + ", " + title

            
            f = train_tmp

            cleaned_identifier = out_ident
            cleaned_body = tagged_body
            new_ident = ""
            new_body = ""
            try:
                f.write(cleaned_identifier + '\n')
            except Exception, e:
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

            inASet = False
            for i in ['train', 'dev', 'test']:
                inASet |= new_ident in splitSets[i] or out_ident in splitSets[i] 

            f.flush()
            ratios =[(correct[i] * 1. / count, int2tags[i+1]) for i in range(len(correct))]
            pprint.pprint(ratios)


    # train_idents = open('/train_ids.p', 'w')
    # dev_idents = open('dev_ids.p', 'w')
    # test_idents = open('test_ids.p', 'w')

    print " IDS", pprint.pprint(idents_split)
    print " Count", count
    print " LEN IDS", len(idents_split)

    pickle.dump(idents_split, open('identifier_to_train_dev_test_partition_and_index.p', 'wb'))
    train.close()
    dev.close()
    test.close()


