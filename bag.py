#Let's view this as a fresh start.
#The first thing is to import the text into a generally usable format.





####Import everything we might need
import pickle, numpy as np, scipy, sklearn, matplotlib, gensim

###Set path
#path = ""
path = 'C:/Users/Glenn/Documents/GitHub/mining-erowid/'

if True:
	dictionary = gensim.corpora.Dictionary.load(path+'erowid_dict.dict')
	corpus = gensim.corpora.MmCorpus(path+'erowid_corpus.mm')	
	experiences = pickle.load(open(path+"experience_index.p","rb"))
	substance_index = pickle.load(open(path+"substance_index.p","rb"))
	tag_index = pickle.load(open(path+"tag_index.p","rb"))


#split the rest off into a different file
#create lists of substances and tags
substances = {}
tags = {}

#convert to NumPy sparse matrices
tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
corpus_matrix = gensim.matutils.corpus2csc(corpus).T
tfidf_matrix = gensim.matutils.corpus2csc(corpus_tfidf).T

##So far, topic modeling has been a disappointment
#lsi = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics=20, id2word=dictionary)
#corpus_lsi = lsi[corpus_tfidf]
#lsi_matrix = gensim.matutils.corpus2csc(corpus_lsi).T