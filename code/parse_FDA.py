from nltk.tokenize import PunktWordTokenizer as WordTokenizer
import random

raw = open('../data/fda.raw', 'r+')
train = open('../data/fda/train.tag', 'w')
dev = open('../data/fda/dev.tag', 'w')
test = open('../data/fda/test.tag', 'w')

int2tags=['TAG', 'food', 'adulterant', 'location', 'year']
generic = ["city", "centre", "county", "street", "road", "and", "in", "town", "village"]

train_cut = .60
dev_cut   = .20
test_cut  = .20

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
                        if cleanEnts(context) == clean_possible_ents:
                            labels.append(j+1)
                            break
            elif len(ent_tokens) > 1:
                cleaned_ent_tokens = cleanEnts(ent_tokens)
                if token in cleaned_ent_tokens:
                    ind = ent_tokens.index(token)
                    context = article[ max(0,i-ind): max(len(article), i + len(ent_tokens)- ind)]
                    if cleanEnts(context) == cleaned_ent_tokens:
                        labels.append(j+1)
            else:
                if token.lower().strip() == ent.lower().strip():
                    labels.append(j+1)
        assert len(labels) <= 1
        label = 0
        if len(labels) == 1:
            label = labels[0]
        tags.append(label)
    return tags

        
tokenizer = WordTokenizer()
scheme = raw.readline()
entry = raw.readline()
limit = 30000000
word_limit = 3000
i = 0
counts = [0] * 5
while not  entry == '':
    if i < limit:
        ents = entry.split('\t')
        ents = ['' if e == 'NULL' else e for e in ents]
        for ent_ind in range(2, len(ents)):
            if ";" in ents[ent_ind]:
                ents[ent_ind] = "|".join([e.strip() for e in ents[ent_ind].split(";")])
        title, article_body, foods, adultrants, locations, dates = ents
        trimmed_article = " ".join(article_body.split(" ")[:min(len(article_body), word_limit)])
        trimmed_article_tokens = tokenizer.tokenize(trimmed_article)
        tagged_body = ""
        gold_ents = [foods, adultrants, locations, dates]
        gold_ents = [e.strip() for e in gold_ents]
        tags = getTags(trimmed_article_tokens, gold_ents)

        ## caclculate statistics on extracted tags
        for k in range(5):
            if k in tags:
                counts[k] += 1
        assert len(trimmed_article_tokens) == len(tags)
        for token, tag in zip(trimmed_article_tokens, tags):
            tagged_body += token + "_" + int2tags[tag] + " "
 
        out_ident = ','.join(gold_ents) + ", " + title
        rand = random.random()
        f = ''
        if rand < train_cut:
            f = train
        elif rand < dev_cut:
            f = dev
        else:
            f = test
        f.write(out_ident + '\n')
        f.write(tagged_body + '\n')

        i+=1
    else:
        break
    entry = raw.readline()
print i

train.flush()
dev.flush()
test.flush()
train.close()
dev.close()
test.close()
