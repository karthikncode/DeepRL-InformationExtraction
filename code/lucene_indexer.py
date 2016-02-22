import sys, pickle, collections, pdb
import lucene
 
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version as Version

def dd():
    return {}

def ddd():
    return collections.defaultdict(dd)

train_articles, train_titles, train_identifiers, train_downloaded_articles, TRAIN_ENTITIES, TRAIN_CONFIDENCES, TRAIN_COSINE_SIM, CONTEXT1, CONTEXT2 = pickle.load(open(sys.argv[1], "rb"))
test_articles, test_titles, test_identifiers, test_downloaded_articles, TEST_ENTITIES, TEST_CONFIDENCES, TEST_COSINE_SIM, CONTEXT1, CONTEXT2 = pickle.load(open(sys.argv[2], "rb"))

articles = train_articles+test_articles
titles = train_titles + test_titles

 
if __name__ == "__main__":
  lucene.initVM()
  indexDir = SimpleFSDirectory(File("index/"))
  # pdb.set_trace()
  writerConfig = IndexWriterConfig(Version.LUCENE_4_9, StandardAnalyzer(Version.LUCENE_4_9))
  writer = IndexWriter(indexDir, writerConfig)
 
  print "%d docs in index" % writer.numDocs()

  for n, l in enumerate(articles):
    text = ' '.join(l[0])
    doc = Document()
    doc.add(Field("text", text, Field.Store.YES, Field.Index.ANALYZED))
    doc.add(Field("title", titles[n], Field.Store.YES, Field.Index.ANALYZED))
    writer.addDocument(doc)
    print "\r", n, '/', len(articles),    
  print "Indexed %d lines from stdin (%d docs in index)" % (n, writer.numDocs())
  print "Closing index of %d docs..." % writer.numDocs()
  writer.close()
