#Let's view this as a fresh start.
#The first thing is to import the text into a generally usable format.





####Import everything we might need

import os, re, pickle, gensim, bs4, nltk

###Set path
path = ""
#path = 'C:/Users/Glenn/Documents/GitHub/mining-erowid/'

###Prepare data-cleaning tools

with open(path+"files/stopwords.xml") as file:
    soup = bs4.BeautifulSoup(file)
    customstop = []
    stopset = soup.find_all('stopword')
    for s in stopset:
        customstop.append(s.contents[0])
    customstop = set(customstop)
    #left in by accident
    customstop.remove("strange")
    
stopwords = nltk.corpus.stopwords.words('english')
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
tagger = nltk.tag.UnigramTagger(nltk.corpus.brown.tagged_sents())
lemmatizer = nltk.WordNetLemmatizer()
from nltk.corpus import wordnet

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
      
def is_okay_word(s):
    import re
    if len(s)==1:
        return False
    elif is_number(s) and float(s)<1900:
        return False
    elif re.match('\d+[mM]?[gGlLxX]',s):
        return False
    elif re.match('\d+[oO][zZ]',s):
        return False
    else:
        return True
      
###Import the experience vault into data structures
substance_index = []
tag_index = []
experiences = os.listdir(path+'xml')

def parse_vault(path):
	for n, experience in enumerate(experiences):
	  #if n>4000:
	    #break
	  with open(path+"xml/"+experience) as f:
	    soup = bs4.BeautifulSoup(f)
	    words = []
	    tokens = tokenizer.tokenize(soup.bodytext.contents[0])
	    pos = tagger.tag(tokens)
	    for token in pos:
	      if token[1] == 'NN':
	        pos = wordnet.NOUN
	      elif token[1] == 'JJ':
	        pos = wordnet.ADJ
	      elif token[1] == 'VB':
	        pos = wordnet.VERB
	      elif token[1] == 'RV':
	        pos = wordnet.ADV
	      else:
	        pos = wordnet.NOUN
	      lemma = lemmatizer.lemmatize(token[0], pos)
	      if is_okay_word(lemma) and lemma not in stopwords and lemma not in customstop:
	        words.append(lemma)
	    substances = [unicode(substance.contents[0]) for substance in soup.find_all("substance")]
	    tags = [unicode(tag.contents[0]) for tag in soup.find_all("tag")]
	    substance_index.append(substances)
	    tag_index.append(tags)
	    
	    if n%1000==0:
	      print("Finished " + str(n) + " files out of " + str(len(experiences)))
	      
	    yield words

#Create gensim data structures
dictionary = gensim.corpora.Dictionary()
corpus = [dictionary.doc2bow(text, allow_update=True) for text in parse_vault(path)]

#serialize
dictionary.save(path+'erowid_dict.dict')
gensim.corpora.MmCorpus.serialize(path+'erowid_corpus.mm', corpus)
pickle.dump(experiences, open(path+"experience_index.p","wb"))
pickle.dump(substance_index, open(path+"substance_index.p","wb"))
pickle.dump(tag_index, open(path+"tag_index.p","wb"))