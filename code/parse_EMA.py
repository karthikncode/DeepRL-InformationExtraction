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
from constants import *

tokenizer = WordTokenizer()

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

def cleanIdent(out_ident, tmp):
    cleaned_identifier = out_ident
    try:
        tmp.write(cleaned_identifier + '\n')
    except Exception, e:
        new_ident = ""
        for c in cleaned_identifier:
            try:
                new_ident += c.encode("ascii", "ignore")
            except Exception, e:
                new_ident += ""
        cleaned_identifier = new_ident;
    return cleaned_identifier

def cleanBody(body, tmp):
    cleanBody = body
    try:
        tmp.write(cleaned_body + '\n')
    except Exception, e:
        new_body = ""
        for c in cleaned_identifier:
            try:
                new_body += c.encode("ascii", "ignore")
            except Exception, e:
                new_body += ""
        cleaned_body = new_body
    return cleanBody




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

        

def writeToFiles(files_dict, write_buffer):
    for file in write_buffer.keys():
        f = files_dict[file]
        write_file = write_buffer[file]
        for i in range(len(write_file)):
            ident = False
            body = False
            data = write_file[i]
            if data == 0:
                ents = ["unknown" for ent in int2tags[1:]]
                title = "skip_article_title"
                ident = ",".join(ents) + ", " + title
                body  = "skip_body"
                f.write(ident + "\n")
                f.write(body + "\n")
            else:
                ident, body = data
            assert body and ident
            f.write(ident + "\n")
            f.write(body + "\n")
        f.flush()
        f.close()

if __name__ == "__main__":

    tmp   = open('../data/tagged_data/EMA2/tmp.3.tag', 'w')
    train = open('../data/tagged_data/EMA2/train.3.tag', 'w') ##This is EMA on the server
    dev = open('../data/tagged_data/EMA2/dev.3.tag', 'w')
    test = open('../data/tagged_data/EMA2/test.3.tag', 'w')

    saveBuffer = pickle.load (open('saveFileToPartitionAndIndex.p','rb'))
    incidents = pickle.load( open('EMA_dump.p', 'rb'))
    downloaded_articles = pickle.load(open('EMA_downloaded_articles_dump.p.server', 'rb'))

    files = {'train': train, 'dev': dev, 'test': test}

    NUM_TRAIN  = 416
    NUM_DEV    = 146
    NUM_TEST   = 124

    write_buffer = {'train':[0]*NUM_TRAIN, 'dev':[0]*NUM_DEV, 'test':[0]*NUM_TEST}
   
    relevant_articles = pickle.load(open('EMA_filtered_articles.p.server', 'rb'))

    ratios = {}
    correct = [0] * (len(int2tags)-1)
    count = 0
    for ind, incident_id in enumerate(incidents.keys()):
        print ind,'/',len(incidents.keys())
        incident = incidents[incident_id]
        if not 'citations' in incident:
            continue
        ents = []
        for ent in int2tags[1:]:
                if ent == "Adulterant":
                    key = "Adulterant(s)"
                else:
                    key = ent
                if key in incident:
                    gold = incident[key].encode('ascii', 'ignore')
                    # print '----------'
                    # print ent, gold
                    gold_split_semi = gold.split(';')                    
                    gold_list = []
                    for semi in gold_split_semi:
                            gold_list += semi.split(',')
                    ents.append("|".join(gold_list))
                else:
                    ents.append('')
        
        for citation_ind, citation in enumerate(incident['citations']):
            title = incident['citations'][citation_ind]['Title']
            saveFile = "../data/raw_data/"+ incident_id+"_"+str(citation_ind)+".raw"
            if not saveFile in relevant_articles or not saveFile in saveBuffer:
                continue
            article = relevant_articles[saveFile]
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

            out_ident = ','.join(ents) + ", " + title
            cleaned_identifier = cleanIdent(out_ident, tmp)

            tagged_body = ""
            for token, tag in zip(tokens, tags):
                try:
                    tagged_body += token + "_" + int2tags[tag] + " "
                except Exception, e:
                    tagged_body += ""
            cleaned_body = cleanBody(tagged_body, tmp)

            outputFiles = saveBuffer[saveFile]
            for file, index in outputFiles:
                write_buffer[file][index] = (cleaned_identifier, cleaned_body)


    ratios =[(correct[i] * 1. / count, int2tags[i+1]) for i in range(len(correct))]
    pprint.pprint(ratios)
    print "Total docs ", count

    for key in write_buffer.keys():
        fails = sum([w == 0 for w in write_buffer[key]])
        print key, "fails",fails


    assert sum([ len(saveBuffer[f]) for f in saveBuffer ]) == sum([len(write_buffer[t]) for t in ["train", "test", "dev"] ])

    writeToFiles(files, write_buffer)

