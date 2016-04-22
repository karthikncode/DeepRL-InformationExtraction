# -*- coding: utf-8 -*-
from __future__ import division
import urllib
import simplejson
# from predict import predict
from scrape import download_article
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib2
import csv
import helper
import time, pickle

# helper.load_constants()

def build_event_dict(csv_file):
    file = open(csv_file,'rb')
    data = csv.reader(file,delimiter=',')
    table = [row for row in data]   
    event_dict = {}
    for i in range(1,len(table)):
        row = table[i]
        article_urls = filter(lambda x: x != '', row[6:])
        metadata = tuple(row[1:6])
        event_dict[metadata] = article_urls
    return event_dict
    
def replace_with_metadata(query_string, meta, metadata):
    replaced_str = query_string
    for i in range(len(meta)):
        replaced_str = replaced_str.replace(meta[i], metadata[i])
    return replaced_str

def test_queries(event_dict):
    '''
    Meta:
    1) date, 2) shooter_name, 3) killed_num, 4) wounded_num, 5) location

    Query types:
        1. (title)
        2. "Shooting in [location]"
        3. "Shooting in [location] on [date]"
        4. "Shooting in [location] on [date], [killed_num] killed"
        5. "Shooting in [location] on [date], [killed_num] killed, [wounded_num] wounded"
        6. "Shooting in [location] on [date] by [shooter name]"
        7. "Shooting in [location] on [date] by [shooter name], [killed_num] killed"
        8. "Shooting in [location] on [date] by [shooter name], [killed_num] killed, [wounded_num] wounded"
        9.
        10.
    '''
    #meta = ['[date]', '[shooter_name]', '[killed_num]', '[wounded_num]', '[location]']
    #query_types = []
    #["Shooting in [location]", "Shooting in [location] on [date]", "Shooting in [location] on [date], [killed_num] killed", 
    #"Shooting in [location] on [date], [killed_num] killed, [wounded_num] wounded", "Shooting in [location] on [date] by [shooter_name]",
    #"Shooting in [location] on [date] by [shooter_name], [killed_num] killed", 
    #"Shooting in [location] on [date] by [shooter_name], [killed_num] killed, [wounded_num] wounded"]
    query_scores = {}
    query_scores_ratios = {}
    count = 0

    for metadata,urls in event_dict.items():
        if urls is None or urls == []:
            continue
        status, title, text, date, title = download_article(urls[0], False, False)
        if title is None or len(title) < 5:
            continue
        print "Event count:",count
        print "Title:",title
        print "Original URL set:",urls
        print
        #urls = set(urls)
        city = metadata[4]
        query_types_with_title = [" ".join([city,title]), " ".join([title,city]), " ".join(title.split()[:10])]#query_types[:]
        query_types_with_title.insert(0, title)
        results = {}
        results_ratios = {}
        for i,query_format in enumerate(query_types_with_title):

            #query = replace_with_metadata(query_format, meta, metadata)
            query = query_format
            #article_urls_google = set(get_related_urls_from_google(query))
            article_urls_bing = get_related_urls_from_bing(query)
            print "Query used:",query
            for url in article_urls_bing:
                print url.encode("ascii","ignore")
            #query_scores[i] = query_scores.get(i,0) + len(article_urls_google.intersection(urls))
            query_scores[i] = query_scores.get(i,0) + count_of_originals(article_urls_bing, urls)
            query_scores_ratios[i] = query_scores_ratios.get(i,0) + count_of_ratio_relevance(article_urls_bing, urls)
            results[i] = query_scores[i]
            results_ratios[i] = query_scores_ratios[i]
            print
        count += 1
        print results
        print results_ratios
        print
    print count
    return query_scores

def count_of_ratio_relevance(searched_urls, original_urls):
    '''
    count of ratios of relevance in search reseult. what percent of results are relevant?
    '''
    if len(searched_urls) == 0: return 0
    number = 0
    for new_url in searched_urls:
        for url in original_urls:
            if url in new_url:
                number += 1
                break
            else:
                match = sum([1 for i,j in zip(url,new_url) if i==j]) / min(len(url),len(new_url))
                if match > 0.95:
                    number +=1
                    break
    return number/len(searched_urls)

def count_of_originals(searched_urls, original_urls):
    '''
    how many of the original articles (~1-3) seen in search results
    '''
    number = 0
    for url in original_urls:
        for new_url in searched_urls:
            if url in new_url:
                print "MATCH:",url,new_url
                number += 1
                break
            else:
                match = sum([1 for i,j in zip(url,new_url) if i==j]) / min(len(url),len(new_url))
                if match > 0.95:
                    print "MATCH:",url,new_url
                    number +=1
                    break
    return number

def number_of_overlaps(searched_urls, original_urls):
    '''
    number of searched urls in original articles set    
    '''
    number = 0 # count an article at most once
    for new_url in searched_urls:
        for url in original_urls:
            if url in new_url:
                print "MATCH:",url,new_url
                number += 1
                break
            else:
                match = sum([1 for i,j in zip(url,new_url) if i==j]) / min(len(url),len(new_url))
                if match > 0.95:
                    print "MATCH:",url,new_url
                    number +=1
                    break
    return number

# def get_predictions_from_query(query_text,original_text,search_engine_name):
#     selected_texts = get_related_articles_from_query(query_text,original_text,search_engine_name)
#     predictions = []
#     for text in selected_texts:
#         predictions.append(predict("trained_model.p", text, False, helper.cities)) # NOTE: Add cities later
#     return predictions

def download_articles_from_query(query_text,original_text,search_engine_name):
    if search_engine_name == 'google':
        article_urls = get_related_urls_from_google(query_text)
    elif search_engine_name == 'bing':
        article_urls = get_related_urls_from_bing(query_text)

    article_texts = []
    article_dates = []
    downloaded_urls = []
    selected_urls = []
    i = 1
    for url in article_urls:
        print "Checking URL ", i
        i+=1
        try:
            if "newslocker" not in url and "newsjs" not in url:
                downloaded_article = download_article(url, False, False)
                article_text = downloaded_article[2]
                article_date = downloaded_article[3]
                article_title = downloaded_article[4]
                if article_text:
                    article_texts.append(article_title + " " + article_text)
                    if article_date != None:
                        article_date = article_date.replace(tzinfo=None)
                    article_dates.append(article_date)
                    downloaded_urls.append(url)
        except Exception, e:
            pass
    for url in downloaded_urls:
        print url.encode("ascii",'ignore')
    #article_texts = filter(None, article_texts)
    article_texts.insert(0,original_text)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(article_texts)
    similarities = similarities_without_duplicates(tfidf_matrix,len(article_texts))#cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()
    print "Similarities", similarities
    similarities_indexed = [(i,x) for i,x in enumerate(similarities) if i > 0] 
    similarities_indexed.sort(key=lambda x:x[1], reverse=True)
    #doc_indices = similarities_indexed[:4] #NOTE: max 8, choose best 4
    doc_indices = similarities_indexed
    selected_texts = []
    selected_dates = []

    #find the true date of the original article
    true_date = None
    for i,x in doc_indices:
        if x > 0.9 and article_dates[i-1] != None:
            true_date = article_dates[i-1]
            break

    #make sure articles are within  a month from the original article.
    for i,x in doc_indices:
        if true_date != None and article_dates[i-1] != None and abs((true_date-article_dates[i-1]).days) > 30:
            continue
        selected_dates.append(article_dates[i-1])
        selected_texts.append(article_texts[i])
        selected_urls.append(downloaded_urls[i-1])


    for i in range(len(selected_urls)):
        print selected_urls[i].encode("ascii",'ignore')
        print selected_dates[i]

    return selected_texts

def similarities_without_duplicates(tfidf_matrix,length):
    include = [1]*length
    similarities_base = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()
    for i in range(length):
        if include[i] == 0:
            continue
        similarities = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix).flatten()
        for j in range(2,length):
            if similarities[j] > 0.98 and j != i:
                include[j] = 0
    for i in range(length):
        if include[i] == 0:
            similarities_base[i] = 0
    return similarities_base


def get_related_urls_from_bing(query_text):
    #BING_API_KEY = 'WvT9Yo4AQHnS7tdCTePQRgRjhHvGOwMJZVjBbMMPSAI' #new key
    # BING_API_KEY = 'q52coTH39rJGmzKKzAeWbrQzDvNIj5OI437Hmwyb5U0' 
    BING_API_KEY = 'Kd/1GUxxE6EPf5H5sX3xL2zS13g0us7HFhjaQ1TKOog' #Yala's key
    credentialBing = 'Basic ' + (':%s' % BING_API_KEY).encode('base64').rstrip()
    searchString = '%27'+urllib.urlencode({"q": query_text})+'%27'

    url = 'https://api.datamarket.azure.com/Bing/Search/v1/Web?' + '$format=json&Query=' + searchString

    request = urllib2.Request(url)
    request.add_header('Authorization', credentialBing)
    response = ""
    count = 0
    while response == "":
        try:
            count += 1
            response = urllib2.urlopen(request) 
        except Exception, e:
            if count > 3:
                return []
            time.sleep(.3)
            pass
    json = simplejson.load(response)
    if json.get('d').get('results') is None:
        print "something wrong"
        return []
    results = json['d']['results']
    article_urls = [item['Url'] for item in results][:20]
    # article_urls = [item['Url'] for item in results][:6]
    return article_urls

def get_related_urls_from_google(query_text):
    GOOGLE_API_KEY = 'AIzaSyBqI9Jtf373tzj_z8LoNX2s5KjIv74-kAg'
    CUSTOM_SEARCH_ENGINE_ID = '002895188139496375183:lzmbhnerau4'
    query = urllib.urlencode({'q': query_text}) # TODO: replace with article title
    url = 'https://www.googleapis.com/customsearch/v1?key=%s&cx=%s&%s' % (GOOGLE_API_KEY,CUSTOM_SEARCH_ENGINE_ID,query)
    results = urllib.urlopen(url)
    json = simplejson.loads(results.read())
    if json.get('items') is None:
        print json
        return []
    results = json['items']
    article_urls = [item['link'] for item in results]
    return article_urls

if __name__ == '__main__':
    event_dict =  build_event_dict('../data/metadata/original/2013MASTER.csv')
    test = test_queries(event_dict)
    print test
