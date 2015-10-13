"""
https://www.jasondavies.com/wordcloud/
create_file("01cannabis", word_chisq("cannabis",data=bdata, n=250,stops=False), binary=False) 
create_file("02mushrooms", word_chisq("mushrooms",data=ldata, n=250,stops=False), binary=False) 
create_file("03salvia", word_chisq("salvia",data=ldata, n=250,stops=False), binary=False) 
create_file("04alcohol", word_chisq("alcohol",data=bdata, n=250,stops=False), binary=True) 
create_file("05mdma", word_chisq("mdma",data=bdata, n=250,stops=False), binary=True) 
create_file("06lsd", word_chisq("lsd",data=ldata, n=250,stops=False), binary=False) 
create_file("07dxm", word_chisq("dxm",data=bdata, n=250,stops=False), binary=True)
create_file("08tobacco", word_chisq("tobacco",data=bdata, n=250,stops=False), binary=True)
create_file("09cocaine", word_chisq("cocaine",data=bdata, n=250,stops=False), binary=True)
create_file("112ci", word_chisq("2ci",data=bdata, n=250,stops=False), binary=True)  

create_file("17dmt", word_chisq("dmt",data=bdata, n=250,stops=False), binary=True)  

create_file("trainwrecks", word_chisq("Train Wrecks & Trip Disasters",data=bdata, n=250,stops=False), binary=True) 
create_file("badtrips", word_chisq("Bad Trips",data=bdata, n=250,stops=False), binary=True)
create_file("mystical", word_chisq("Mystical Experiences",data=bdata, n=250,stops=False), binary=True)
create_file("glowing", word_chisq("Glowing Experiences",data=bdata, n=250,stops=False), binary=True)
create_file("difficult", word_chisq("Difficult Experiences",data=bdata, n=250,stops=False), binary=True)


10 amphetamines
11 2-CI
12 Monrning Glory
13 Nitrous
14 Syrian Rue
15 Meth
16 5-MEO-DMT
17 DMT
18 Ketamine
19 5-MEO-DIPT



word_chisq("lsd",reference=[k for k in substance_count.keys())


"""

if __name__ == "__main__":
	path = 'C:/Users/Glenn/Documents/GitHub/ineffable/'
	import os, re, pickle, bs4, nltk, random, numpy as np
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.preprocessing import normalize
	vectorizer = pickle.load(open(path+"data/pickle/vectorizer.p","rb"))
	bowdata = pickle.load(open(path+"data/pickle/data.p","rb"))
	ndata = 1000*normalize(bowdata.astype(np.float), norm='l1',axis=1)
	ldata = bowdata.copy()
	ldata.data = np.log(bowdata.data+1)
	lndata = 1000*normalize(ldata.astype(np.float), norm='l1',axis=1)
	bdata = bowdata.astype(bool)
	experiences = pickle.load(open(path+"data/pickle/experience_index.p","rb"))
	substance_index = pickle.load(open(path+"data/pickle/substance_index.p","rb"))
	tag_index = pickle.load(open(path+"data/pickle/tag_index.p","rb"))
	substance_count = pickle.load(open(path+"data/pickle/substance_count.p","rb"))
	tag_count = pickle.load(open(path+"data/pickle/tag_count.p","rb"))
	all_tags = sorted(tag_count.keys() + substance_count.keys())
	all_index = [tag_index[n] + substance_index[n] for n,row in enumerate(experiences)]
	vocab = np.array(vectorizer.get_feature_names())
	with open(path+"data/files/stopwords.txt") as file:
		substops = set([line.replace("\n","") for line in file.readlines()])
		
	def word_chisq(	key,
					reference=None,
					data=ndata,
					vocab=vocab,
					n=10,
					stops=False):
		from sklearn.feature_selection import chi2
		if type(key) == str:
				key = [key]

		if reference == None:
			refdata = data
			refs = [True for row in all_index]
			refs = np.array(refs)
		else:
			if type(reference) == str:
				reference = [reference]
				
			refs = [(True in [(k in row) for k in reference]) for row in all_index]
			refs = np.array(refs)
			refdata = data[refs,:]

		#okay...so we're gettin' there...refdata now works correctly, but labels is wrong
		labels = [(True in [(k in row) for k in key]) for row in all_index]
		labels = np.array(labels)
		labels = labels[refs]

		chisq, p = chi2(refdata, labels)
		ranking = np.argsort(chisq)[::-1]
		values = []
		freqs = (refdata > 0)[labels,:].sum(axis=0)
		
		i = 0
		for rank in ranking:
			if i >= n:
				break
				
			if key in substance_count.keys() and vocab[rank] in substops and stops==True:
				continue
			elif not np.isnan(chisq[rank]):
				values.append((vocab[rank],chisq[rank],p[rank],freqs[:,rank][0,0]))
				i+=1
				
		return values[0:n]
		
		
	y,n,x = "y","n","x"
	def create_file(key, values, suffix="", filter=True, binary=False, nwords=50): 
		filename = key+suffix
		filename+=".txt"
		
		print "Building " +  filename + ":"
		with open(path+"master/output/" + filename,"w") as file:
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
					
	def examples(word, lst=None, sort=True, n=5):
		sub, _, _, _, subexps = rowslice(lst)
		wdata = sub[:,vocab==word].toarray()
		ranking = np.argsort(wdata[:,0])[::-1]
		exps = []
		for rank in ranking:
			if wdata[rank,0] > 0:
				exps.append(subexps[rank])
				
		if sort==False:
			exps = sorted(exps, key=lambda *args: random.random())
			
		return exps[0:n]
		
	def read_examples(word, lst=None, sort=True, n=5):
		files = examples(word, lst, sort, n)
		print "Word occurs in " + str(len(files)) + " files. (note this is wrong!)"
		for name in files:
			dummy = raw_input("*************Next report?**************")
			with open(path+"data/xml/"+name) as f:	
				txt = f.read()
				txt = txt.replace("*","")
				txt = txt.replace(word,"***"+word+"***")
				txt = txt.replace(word.capitalize(),"***"+word.capitalize()+"***")
				print txt
		
		
	def tag_chisq(key, n=10):
		from sklearn.feature_selection import chi2
		if key in all_tags:
			labels = [(key in row) for row in all_index]
		else:
			raise ValueError('Not a valid tag or substance')
			
		mat = np.zeros((len(experiences),len(all_tags)))
		for i,row in enumerate(experiences):
			for j,tag in enumerate(all_tags):
				if tag in all_index[i]:
					mat[i,j] = True
				else:
					mat[i,j] = False
			
		chisq, p = chi2(mat, labels)
		ranking = np.argsort(chisq)[::-1]
		values = []
		for rank in ranking:
			values.append((chisq[rank],all_tags[rank],p[rank]))
		return values[0:n]
						
				
				
	def word_mannwhitney(key, data=ndata, vocab=vocab, index=all_index, method="raw", n=10, stops=True):
		if key in all_tags:
			labels = [(key in row) for row in index]
		else:
			raise ValueError('Not a valid tag or substance')
			
		from scipy.stats import mannwhitneyu
		mannwhit = []
		for i in range(data.shape[1]):
			ranks = data[:,i].toarray().argsort(axis=0)[::-1]
			trues = ranks[np.asarray(labels)==True]
			falses = ranks[np.asarray(labels)==False]
			u = mannwhitneyu(trues,falses)[0]
			p = u/(trues.shape[0]*falses.shape[0])
			mannwhit.append(p)
			
		ranking = np.argsort(mannwhit)[::-1]	
		values = []
		for rank in ranking:
			if key in substance_count.keys() and vocab[rank] in substops and stops==True:
				continue
			else:
				values.append((mannwhit[rank],vocab[rank]))
		return values[0:n]
		
	def word_tfidf(key, data=bdata, vocab=vocab, n=10, mindocs=5,stops=True):
		tf = rowslice(key,data=data)[0].sum(axis=0)
		idf = 1.0/data.sum(axis=0)
		tfidf = np.multiply(tf,idf).tolist()[0]
		ranking = np.argsort(tfidf)[::-1]
		tflist = tf.tolist()[0]
		dflist = (1.0/idf).tolist()[0]
		values = []
		for rank in ranking:
			if key in substance_count.keys() and vocab[rank] in substops and stops==True:
				continue
			#elif tfidf[rank] < 1:
			elif dflist[rank] > mindocs:
			#else:
				values.append((tfidf[rank],vocab[rank],tflist[rank],dflist[rank]))
	
		return values[0:n]
		
		
	def sample(data=ndata, rows=1000):
		shuffle = range(data.shape[0])[0:rows]
		return data[sorted(shuffle, key=lambda *args: random.random()),:]
		
		

		
		
	def exclude_words(data=bowdata, lst=[]):
		from scipy.sparse import csr_matrix, lil_matrix
		excluded = lil_matrix(data)
		for i,word in enumerate(vocab):
			if word in lst:
				excluded[:,i] = np.zeros((data.shape[0],1))
		return csr_matrix(excluded)
		
	
	def word_sims(data=ndata, words=1000):
		from sklearn.metrics.pairwise import cosine_similarity
		freqs = bowdata.sum(axis=0)
		ranking = np.argsort(freqs).tolist()[0][::-1]
		usewords = vocab[ranking]
		usewords = usewords[0:words]
		wdata = data[:,ranking]
		wdata = wdata.T
		wdata = wdata[0:words,:]
		similarities = cosine_similarity(wdata)
		return similarities, usewords
		

	def top_words(data=bowdata, words=25000):
		freqs = data.sum(axis=0)
		ranking = np.argsort(freqs).tolist()[0][::-1]
		usewords = vocab[ranking]
		usewords = usewords[0:words]
		wdata = data[:,ranking]
		wdata = wdata[:,0:words]
		return wdata, usewords
		
	
	tdata,usewords = top_words()
	mw = word_mannwhitney("lsd",data=tdata,vocab=usewords)
	
	def exp_sims(data=ndata, words=1000):
		from sklearn.metrics.pairwise import cosine_similarity
		freqs = bowdata.sum(axis=0)
		ranking = np.argsort(freqs).tolist()[0][::-1]
		usewords = vocab[ranking]
		usewords = usewords[0:words]
		wdata = data[:,ranking]
		wdata = wdata[:,0:words]
		similarities = cosine_similarity(wdata)
		return similarities, usewords
		
		
	def cluster(sims):
		from scipy.cluster.hierarchy import ward, to_tree
		r = ward(sims)
		t = to_tree(r)
		return r, t
		
	
	def recurse(tree, vocb):
		if tree.is_leaf():
			return vocb[tree.id]
		left = tree.get_left()
		right = tree.get_right()
		return (recurse(left,vocb),recurse(right,vocb))
        
        		
	def group_tags(data=bowdata):
		groups = np.zeros((0,data.shape[1]))
		for tag in all_tags:
			bools = [tag in row for row in all_index]
			row = data[np.asarray(bools),:].sum(axis=0)
			groups = np.vstack((groups,row))
		return groups
		
		
w, usewords = word_sims()
c, t = cluster(w)
r = recurse(t,usewords)

		
def jsontree(tree, parent, vocb):
		node = {}
		if tree.is_leaf():
			node["name"] = vocb[tree.id]
		else:
			node["name"] = str(tree.id)		
			left = tree.get_left()
			right = tree.get_right()
			node["children"] = [jsontree(left,str(tree.id),vocb),jsontree(right,str(tree.id),vocb)]
		node["parent"] = parent
		return node

with open(path+"tree.json","wb") as j:
	import json
	json.dump(jsontree(t,None,usewords),j)
	

def word_blend(key, data=bdata, blend=0.5, index=all_index, vocab=vocab, n=10, mindocs=5,stops=True):
	from sklearn.feature_selection import chi2
	if key in all_tags:
		labels = [(key in row) for row in index]
	else:
		raise ValueError('Not a valid tag or substance')
	#calculate chisq
	chisq, p = chi2(data, labels)
	#calculate tf-idf
	
	squared = np.multiply(chisq,tfidf)
	root = np.sqrt(squared)
	ranking = np.argsort(root)[::-1]
	values = []
	for rank in ranking:
		if key in substance_count.keys() and vocab[rank] in substops and stops==True:
			continue
		elif dflist[rank] > mindocs:
			values.append((root[rank],vocab[rank]))
	
	return values[0:n]

#doesn't seem to work quite right..."subtract" totally dominates?
def word_blend(key, data=bdata, p=0.5, index=all_index, vocab=vocab, n=10, mindocs=5,stops=True):
	group = rowslice(key,data=data)[0]
	n1 = float(group.shape[0])
	ingroup = group.sum(axis=0).astype(float)
	allgroup = data.sum(axis=0).astype(float)
	n2 = float(data.shape[0])
	divide = (n2/n1)*np.divide(ingroup,allgroup)
	subtract = (ingroup/n1) - (allgroup/n2)
	subtract = np.maximum(np.zeros_like(subtract),subtract)
	combined = np.zeros_like(divide)
	for i in range(combined.shape[1]):
		combined[0,i] = angle_blend(divide[0,i],subtract[0,i],p)
		
	combined = combined.tolist()[0]
	ingroup = ingroup.tolist()[0]
	ranking = np.argsort(combined)[::-1]
	values = []
	for rank in ranking:
		if key in substance_count.keys() and vocab[rank] in substops and stops==True:
			continue
		elif ingroup[rank] > mindocs:
			values.append((combined[rank],vocab[rank]))
	
	return values[0:n]
	

	
def angle_blend(x,y,p):
	import math
	if p<0 or p>1:
		print "invalid value"
		return None
       
	a = p*math.pi/2.0
	b = math.atan2(y,x)
	if a == b:
		return math.sqrt(x*x+y*y)
	elif a > b:
		return y/math.cos(a)
	elif a < b:
		return x/math.sin(a)


def word_custom(key, k=0, data=ndata, vocab=vocab, index=all_index, method="raw", n=10, stops=True):
		from sklearn.feature_selection import chi2
		if key in all_tags:
			labels = [(key in row) for row in index]
		else:
			raise ValueError('Not a valid tag or substance')
			
		
		for word in vocab:
			
		chisq, p = chi2(data, labels)
		ranking = np.argsort(chisq)[::-1]
		values = []
		
		counts = rowslice(key,data=bdata)[0]
		freqs = counts.sum(axis=0)
		
		for rank in ranking:
			if key in substance_count.keys() and vocab[rank] in substops and stops==True:
				continue
			else:
				values.append((chisq[rank],vocab[rank],p[rank],freqs[:,rank][0,0]))
		return values[0:n]
		
		
		
def distinctive(X, y, _k):
	import scipy
	from sklearn.utils import (as_float_array, check_array, check_X_y, safe_sqr, safe_mask)
	from sklearn.preprocessing import LabelBinarizer
	from scipy.sparse import issparse
	from sklearn.utils.extmath import norm, safe_sparse_dot
	# XXX: we might want to do some of the following in logspace instead for
	# numerical stability.
	X = check_array(X, accept_sparse='csr')
	if np.any((X.data if issparse(X) else X) < 0):
		raise ValueError("Input X must be non-negative.")
	
	Y = LabelBinarizer().fit_transform(y)	
	if Y.shape[1] == 1:
		Y = np.append(1 - Y, Y, axis=1)
	
	observed = safe_sparse_dot(Y.T, X)  # n_classes * n_features
	
	feature_count = check_array(X.sum(axis=0))
	class_prob = check_array(Y.mean(axis=0))
	expected = np.dot(class_prob.T, feature_count)
	
	observed = np.asarray(observed, dtype=np.float64)
	k = len(observed)
	chisq = observed
	chisq -= expected
	chisq **= 2
	chisq /= expected**(_k+1)
	chisq = chisq.sum(axis=0)
	return chisq, scipy.special.chdtrc(k - 1, chisq)
	    
    
def word_distinct(key, k=1, data=ndata, vocab=vocab, index=all_index, method="raw", n=10, stops=True):
		from sklearn.feature_selection import chi2
		if key in all_tags:
			labels = [(key in row) for row in index]
		else:
			raise ValueError('Not a valid tag or substance')
			
		chisq, p = distinctive(data, labels,k)
		ranking = np.argsort(chisq)[::-1]
		values = []
		
		counts = rowslice(key,data=bdata)[0]
		freqs = counts.sum(axis=0)
		
		for rank in ranking:
			if key in substance_count.keys() and vocab[rank] in substops and stops==True:
				continue
			else:
				values.append((chisq[rank],vocab[rank],p[rank],freqs[:,rank][0,0]))
		return values[0:n]