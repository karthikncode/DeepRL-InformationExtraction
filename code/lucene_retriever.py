import sys
import lucene
 
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
 
if __name__ == "__main__":
    lucene.initVM()
    analyzer = StandardAnalyzer(Version.LUCENE_4_9)
    reader = IndexReader.open(SimpleFSDirectory(File("index/")))
    searcher = IndexSearcher(reader)

    while True:
        input_query = raw_input("Enter Query: ") 
        print input_query

        query = QueryParser(Version.LUCENE_4_9, "text", analyzer).parse(input_query)
        # query = QueryParser(Version.LUCENE_4_9, "text", analyzer).parse("title:"+input_query+" OR text: "+input_query)
        # query = MultiFieldQueryParser({"text", "title"}, analyzer).parse("title:'"+input_query+"'' OR text:'"+input_query+"'")
        MAX = 10
        hits = searcher.search(query, MAX)
     
        print "Found %d document(s) that matched query '%s':" % (hits.totalHits, query)
        for hit in hits.scoreDocs:
            print hit.score, hit.doc, hit.toString()
            doc = searcher.doc(hit.doc)
            print doc.get("title").encode("utf-8")
