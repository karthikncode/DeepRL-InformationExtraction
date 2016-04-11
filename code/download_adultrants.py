''' download all articles referenced in EMA scrape'''

#trainFile = '../data/tagged_data/whole_text_full_city/train.tag'

import sys, pickle, pdb
import query as query


if __name__ == '__main__':
    incidents = pickle.load(open('EMA_dump.p', 'rb'))
    saveFile = "EMA"
    
    # downloaded_articles = {}
    downloaded_articles = pickle.load(open('EMA_downloaded_articles_dump.p', 'rb'))

    for incident_ind, incident_id in enumerate(incidents.keys()):
        incident = incidents[incident_id]
        summary = incident['Incident_Summary']
        if not 'citations' in incident:
            continue
        citations = incident['citations']
        for citation_ind, citation in enumerate(citations):
            saveFile = "../data/raw_data/"+ incident_id+"_"+str(citation_ind)+".raw"
            title = citation['Title']
            source = citation['Source']
            print "Now downloading ", incident_ind, "/", len(incidents), "pt", citation_ind
            if saveFile in downloaded_articles:
                print saveFile, "skipped"
                continue
            # try:
            with open(saveFile, "wb" ) as f:
                query_text = title+' '+ source
                clean_query = query_text.encode("ascii", "ignore")
                clean_summary= summary.encode("ascii", "ignore")
                articles = query.download_articles_from_query(clean_query, clean_summary,'bing')
                if len(articles) > 0:
                    article = articles[0]
                    f.write(article)
                    f.flush()
                    f.close()
                    downloaded_articles[saveFile] = article
                    pickle.dump(downloaded_articles, open('EMA_downloaded_articles_dump.p', 'wb'))
                else:
                    downloaded_articles[saveFile] = "None"
                    pickle.dump(downloaded_articles, open('EMA_downloaded_articles_dump.p', 'wb'))
            # except Exception, e:
            #     downloaded_articles[saveFile] = "None"
            #     pickle.dump(downloaded_articles, open('EMA_downloaded_articles_dump.p', 'wb'))
            #     # raise e
            print "Saved to file", saveFile
        print
    #save to file
    pickle.dump(downloaded_articles, open('EMA_downloaded_articles_dump.p', 'wb'))
    
