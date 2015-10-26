"""
#serialize
pickle.dump(vectorizer, open(path+"vectorizer.p","wb"))
pickle.dump(data, open(path+"data.p","wb"))
pickle.dump(experiences, open(path+"experience_index.p","wb"))
pickle.dump(substance_index, open(path+"substance_index.p","wb"))
pickle.dump(tag_index, open(path+"tag_index.p","wb"))
pickle.dump(substance_count, open(path+"substance_count.p","wb"))
pickle.dump(tag_count, open(path+"tag_count.p","wb"))

#unserialize
if True:
	import os, re, pickle, bs4, nltk, numpy as np
	from sklearn.feature_extraction.text import CountVectorizer
	path = 'C:/Users/Glenn/Documents/GitHub/mining-erowid/'
	vectorizer = pickle.load(open(path+"vectorizer.p","rb"))
	data = pickle.load(open(path+"data.p","rb"))
	experiences = pickle.load(open(path+"experience_index.p","rb"))
	substance_index = pickle.load(open(path+"substance_index.p","rb"))
	tag_index = pickle.load(open(path+"tag_index.p","rb"))
	substance_count = pickle.load(open(path+"substance_count.p","rb"))
	tag_count = pickle.load(open(path+"tag_count.p","rb"))
	vocab = np.array(vectorizer.get_feature_names())
	with open(path+"files/stopwords.txt") as file:
		substops = set([line.replace("\n","") for line in file.readlines()])
"""

#Let's view this as a fresh start.
#The first thing is to import the text into a generally usable format.





####Import everything we might need

import os, re, pickle, bs4, nltk, numpy as np
from sklearn.feature_extraction.text import CountVectorizer

###Set path
#path = ""
path = 'C:/Users/Glenn/Documents/GitHub/mining-erowid/'



##Prepare data-cleaning helper functions
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
substance_index, tag_index, experiences = [], [], os.listdir(path+'xml')

def parse_vault(path):
	###Prepare data cleaning tools
	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	tagger = nltk.tag.UnigramTagger(nltk.corpus.brown.tagged_sents())
	lemmatizer = nltk.WordNetLemmatizer()
	from nltk.corpus import wordnet
	german = nltk.corpus.stopwords.words('german')
	
	for n, experience in enumerate(experiences):
		#if n>10:
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
				if is_okay_word(lemma):
					words.append(lemma)
	
			substances = [unicode(substance.contents[0]) for substance in soup.find_all("substance")]
			tags = [unicode(tag.contents[0]) for tag in soup.find_all("tag")]
		            
			substance_index.append(substances)
			tag_index.append(tags)
	
			yield " ".join(words)
			if n%1000==0:
				print("Finished " + str(n) + " files out of " + str(len(experiences)))

##create sklearn data structures
vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english')) 
#vectorizer = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=25) 
data = vectorizer.fit_transform(parse_vault(path))
vocab = np.array(vectorizer.get_feature_names())

from collections import Counter
tag_count, substance_count = Counter(), Counter()
for row in tag_index:
	for tag in row:
		tag_count[tag]+=1
		
for row in substance_index:
	for substance in row:
		substance_count[substance]+=1		

###Prepare custom stopwords
with open(path+"files/stopwords.txt") as file:
	substops = set([line.replace("\n","") for line in file.readlines()])
   
#with open(path+"files/stopwords.txt","w") as file:
#	for word in sorted(substops):
#		file.write(str(word))
#		file.write("\n")
		

	
		
def word_chisq(key, n=10, stops=True):
	from sklearn.feature_selection import chi2
	if key in tag_count.keys():
		labels = [(key in row) for row in tag_index]
	elif key in substance_count.keys():
		labels = [(key in row) for row in substance_index]
		
	chisq, p = chi2(data, labels)
	ranking = np.argsort(chisq)[::-1]
	values = []
	for rank in ranking:
		if key in substance_count.keys() and vocab[rank] in substops and stops==True:
			continue
		else:
			values.append((chisq[rank],vocab[rank],p[rank]))
	return values[0:n]

def bool_chisq(key, n=10, stops=True):
	from sklearn.feature_selection import chi2
	if key in tag_count.keys():
		labels = [(key in row) for row in tag_index]
	elif key in substance_count.keys():
		labels = [(key in row) for row in substance_index]
		
	chisq, p = chi2(data>0, labels)
	ranking = np.argsort(chisq)[::-1]
	values = []
	for rank in ranking:
		if key in substance_count.keys() and vocab[rank] in substops and stops==True:
			continue
		else:
			values.append((chisq[rank],vocab[rank],p[rank]))
	return values[0:n]


y,n,x = "y","n","x"
def create_file(key, binary=False, filter=True,nwords=50):
	from sklearn.feature_selection import chi2
	if key in tag_count.keys():
		labels = [(key in row) for row in tag_index]
	elif key in substance_count.keys():
		labels = [(key in row) for row in substance_index]
	
	if binary==False:
		chisq, p = chi2(data, labels)
	else:
		chisq, p = chi2(data>0, labels)
	ranking = np.argsort(chisq)[::-1]
	values = []
	for rank in ranking:
		values.append((chisq[rank],vocab[rank],p[rank]))
	
	filename = key
	if binary==True:
		filename+="_bin"
	if filter==False:
		filename+="_nof"
	filename+=".txt"
	
	print "Building " +  filename + ":"
	with open(path+"output/" + filename,"w") as file:
		j = 0
		for value in values:
			if j!=None and j>nwords:
				return
				
			if filter==True:
				response = input("Use " + str(value) + "? (" + str(j) + " words so far) (y/n/x)")
			else:
				response = "y"
			
			if response == "y":	
				if binary==True:
					r = int(value[0]/10)
				else:
					r = int(value[0]/100)
				for i in range(r):
					file.write(value[1])
					file.write(" ")
				print "Wrote " + str(value[1]) + " " + str(r) + " times."
				j+=1
			elif response == "x":
				print "Finished " + filename + "."
				return
			else:
				continue

#Subset the data by tag and/or substance
def dataslice(lst):
	if lst==None:
		return data, tag_index, substance_index, experiences
		
	if type(lst)==str:
		lst = [lst]
	indices = []
	slice_tags = []
	slice_substances = []
	slice_exps = []
	for n,exp in enumerate(experiences):
		for item in lst:
			if item in tag_index[n] or item in substance_index[n]:
				indices.append(n)
				slice_tags.append(tag_index[n])
				slice_substances.append(substance_index[n])
				slice_exps.append(experiences[n])
				break
	return data[indices], slice_tags, slice_substances, slice_exps
	
#This is sort of a bad setup; instead I should pass ar eference to the data
def slice_chisq(d, tindex, sindex, key, n=10, stops=True):
	from sklearn.feature_selection import chi2
	if key in tag_count.keys():
		labels = [(key in row) for row in tindex]
	elif key in substance_count.keys():
		labels = [(key in row) for row in sindex]
		
	chisq, p = chi2(d, labels)
	ranking = np.argsort(chisq)[::-1]
	values = []
	for rank in ranking:
		if key in substance_count.keys() and vocab[rank] in substops and stops==True:
			continue
		elif not np.isnan(chisq[rank]):
			values.append((chisq[rank],vocab[rank],p[rank]))
	return values[0:n]
			
def word_index(word):
	return np.where(vocab==word)[0][0]
	
def examples(word, lst=None, sort=True, n=5):
	sub, foo, bar, subexps = dataslice(lst)
	wdata = sub[:,vocab==word].toarray()
	ranking = np.argsort(wdata[:,0])[::-1]
	exps = []
	for rank in ranking:
		if wdata[rank,0] > 0:
			exps.append(subexps[rank])
			
	if sort==False:
		exps = sorted(exps, key=lambda *args: random.random())
		
	return exps[0:n] 

def readfile(exp):
	with open(path+"xml/"+exp) as f:
		print f.read()
		
		
def read_examples(word, lst=None, sort=True, n=5):
	files = examples(word, lst, sort, n)
	print "Word occurs in " + str(len(files)) + " files."
	for name in files:
		dummy = raw_input("*************Next report?**************")
		with open(path+"xml/"+name) as f:	
			txt = f.read()
			txt = txt.replace("*","")
			txt = txt.replace(word,"***"+word+"***")
			txt = txt.replace(word.capitalize(),"***"+word.capitalize()+"***")
			print txt