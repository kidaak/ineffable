"""
https://www.jasondavies.com/wordcloud/
curate("cannabis",word_chisq("cannabis",data=ldata, n=250),jsonize=True)
	curate("mushrooms",word_chisq("mushrooms",data=ldata, n=250),jsonize=True)
	curate("salvia",word_chisq("salvia",data=ldata, n=250),jsonize=True)
	curate("alcohol",word_chisq("alcohol",data=ldata, n=250),jsonize=True)
	curate("mdma",word_chisq("mdma",data=ldata, n=250),jsonize=True)
	curate("lsd",word_chisq("lsd",data=ldata, n=250),jsonize=True)
	curate("dxm",word_chisq("dxm",data=ldata, n=250, minwords=25),jsonize=True)
	curate("tobacco",word_chisq("tobacco",data=ldata, n=250, minwords=25),jsonize=True)
	curate("cocaine",word_chisq("cocaine",data=ldata, n=250),jsonize=True)
	curate("nitrous",word_chisq("nitrous",data=ldata, n=250, minwords=25),jsonize=True)
	curate("dmt",word_chisq("dmt",data=ldata, n=250, minwords=25),jsonize=True)
	curate("meth",word_chisq("meth",data=ldata, n=250, minwords=25),jsonize=True)
	curate("amphetamines",word_chisq("amphetamines",data=ldata, n=250, minwords=25),jsonize=True)
	curate("ketamine",word_chisq("ketamine",data=ldata, n=250, minwords=25),jsonize=True)
	curate(["datura","brugmansia"],word_chisq(("datura","brugmansia"),data=ldata, n=250, minwords=25),jsonize=True)
	curate("2cb",word_chisq("2cb",data=ldata, n=250, minwords=25),jsonize=True)
	curate("kratom",word_chisq("kratom",data=ldata, n=250, minwords=25),jsonize=True)
	curate("2ci",word_chisq("2ci",data=ldata, n=250, minwords=25),jsonize=True)
	curate("syrian_rue",word_chisq("syrian_rue",data=ldata, n=250, minwords=25),jsonize=True)
	curate("5meo_dmt",word_chisq("5meo_dmt",data=ldata, n=250, minwords=25),jsonize=True)

	cloud(word_chisq(("datura","brugmansia"),data=ldata, n=50, minwords=25))
	cloud(word_chisq(("kratom"),data=ldata, n=50, minwords=25))
	cloud(word_chisq(("2ci"),data=ldata, n=50, minwords=25),jsonize=True)
	cloud(word_chisq(("syrian_rue"),data=ldata, n=50, minwords=25),jsonize=True)

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

[u'ketamine', u'lsd', u'nitrous', u'alcohol', u'tobacco', u'mdma', u'amphetamines', u'brugmansia', u'salvia', u'mushrooms',
 u'cannabis', u'dxm', u'dmt', u'kratom', u'meth', u'2cb', u'datura']
categories:
	- nootropics and herbs and supplements
	- stimulants
	- depressants
	- entheogens
	- short-acting entheogens
	- entactogens
	- deleriants
	- dissociatives
	- tobacco
	- alcohol
	- cannaboids
	- PIHKAL
	- TIHKAL
	- natural
	- synthetic
	- short-acting
	- long-acting
	- opiates
	- barbiturates
	- benzodiazepines
	- medications


"""

if __name__ == "__main__":
	#path = 'C:/Users/Glenn/Documents/GitHub/ineffable/'
	path = 'C:/Users/Glenn Wright/Documents/GitHub/ineffable/'
	import os, re, pickle, bs4, nltk, random, numpy as np, scipy as sp, json
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
	all_count = {}
	for key in substance_count.keys():
		all_count[key] = substance_count[key]
	for key in tag_count.keys():
		all_count[key] = substance_count[key]
	all_tags = sorted(tag_count.keys() + substance_count.keys())
	subs50 = dict([(key,substance_count[key]) for key in substance_count if substance_count[key]>=50])
	subs100 = dict([(key,substance_count[key]) for key in substance_count if substance_count[key]>=100])
	all_index = [tag_index[n] + substance_index[n] for n,row in enumerate(experiences)]
	vocab = np.array(vectorizer.get_feature_names())
	with open(path+"data/customstops.json","rb") as f:
		customstops = json.loads(f.read())
	with open(path+"/master/ENGSTOP.csv",'rb') as csvfile:
		import csv
		reader = csv.reader(csvfile)
		extrastops = [row[0].replace("'","") for row in reader][1:]
	wordcounts = dict([(vocab[i],n) for i,n in enumerate(bowdata.sum(axis=0).tolist()[0])])

	def reduce_data(data, minwords=50):
		#drop rare words
		freqs = (data > 0).sum(axis=0)
		mask = []
		v = []
		for i in range(data.shape[1]):
			if freqs[0,i] > minwords:
				mask.append(i)
				v.append(vocab[i])
		cdata = data[:,mask]
		stops = set()
		#remove every substance-specific stopword
		for k in customstops.keys():
			for word in customstops[k]:
				stops.add(word)
		#also remove the names of substances, because some might not have been curated yet
		for s in substance_count.keys():
			stops.add(s)
		bool = [word not in stops for word in v]
		mask = []
		v2 = []
		for i in range(len(bool)):
			if bool[i]:
				mask.append(i)
				v2.append(v[i])
		cdata = cdata[:,mask]
		return cdata, v2

if True:
	import gensim
	reduced, v = reduce_data(bowdata)
	corpus = gensim.matutils.Sparse2Corpus(reduced.T)
	tfidf = gensim.models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]
	dv = dict([(k,vc) for k,vc in enumerate(v)])
 	#passes=2, iterations=100?
	lda = gensim.models.LdaModel(corpus_tfidf, alpha='auto', id2word = dv, num_topics = 50, iterations=200, passes=2, eta=1.0/len(dv))
	corpus_lda= lda[corpus_tfidf]
	lda.print_topics()

    lda_data = gensim.matutils.corpus2csc(corpus_lda).T
	topdocs = {}
   	for m,topic in enumerate(lda.print_topics()):
        top = np.argmax(lda_data[:,m].toarray())
        topdocs[topic] = experiences[top]


def view_reports(lst):
	if type(lst)==str:
		lst = [lst]
	print type(lst)
	import webbrowser
	webbrowser.open_new('file:///'+path+'data/xml/'+lst[0])
	#webbrowser.open_new('file://'+path+'data/xml/' + lst[0])
	from time import sleep
	for item in lst[1:]:
		sleep(1)
		webbrowser.open_new_tab('file:///'+path+'data/xml/' + item)



def topic_exemplars(data, n, ndocs=5):
	import webbrowser
	data = data.toarray()
	t = np.argsort(data[:,n].tolist())
	exps = [experiences[i] for i in t[-1:-ndocs-1:-1]]
	return exps

def prepare_eta(vocab, lst, num_topics=50):
	et = 1.0/len(vocab)
	eta = np.linspace(et,et,num_topics*len(vocab))
	eta = eta.reshape((num_topics, len(vocab)))
	#apply weights
	for n,l in enumerate(lst):
		for word in l:
			if word in vocab:
				idx = vocab.index(word)
				eta[n,idx] = 100.0*et
			else:
				print "note: skipped " + word
	return eta

weights = [["police","handcuff","cop","arrest"],["overdose","narcan","hospital","nurse","ambulance"]]
eta = prepare_eta(v,weights)

	hdp = gensim.models.HdpModel(corpus_tfidf, id2word = dv)
	corpus_hdp = hdp[corpus_tfidf]
	hdp.print_topics()
	#some way of seeding this?

#this might work for seeded LDA
if True:
	seedterms = {
		"seedterm1" : ["addict","addiction","problem"],
		"seedterm2" : ["psychedelic","fractal","visual"]
	}
	seedvocab = vocab + seedterms.keys()
	newcols = np.zeros((bowdata.shape[0],len(seedterms.keys())))
	for n, row in enumerate(bowdata):
		for m,term in enumerate(seedterms.keys()):
				found = False
				for word in seedterms[term]:
					if bowdata[n, vocab.index(word)] > 0:
						found = True
				if found:
					newcols[n,m] = 1
	xdata = np.concatenate((bowdata,newcols),axis=1)



	def summ_subs(data):
		all_subs = sorted(substance_count.keys())
		summs = np.zeros((len(all_subs),data.shape[1]))
		for y,tag in enumerate(all_subs):
			print "working on " + tag
			mask = []
			for e, exp in enumerate(all_index):
				if tag in exp:
					mask.append(e)
			tdata = data[mask, :]
			summ = tdata.mean(axis=0)
			summs[y] = summ
		return summs

	def similarity(data):
		from sklearn.metrics.pairwise import cosine_similarity
		sims = cosine_similarity(data)
		return sims




if True:
	reduced, v = reduce_data(bdata)
	simplified = summ_subs(reduced)
	similar = similarity(simplified)
	from scipy.cluster.hierarchy import ward
	clusters = ward(similar)
	subnames = sorted(substance_count.keys())
	subcounts = [substance_count[key] for key in subnames]

if True:
	tree = jsontree(clusters,2*clusters.shape[0],subnames,subcounts,1,np.nan)
	#tree = jsontree(clusters,2*clusters.shape[0],subnames,subcounts,np.nan,1000)
	with open(path+"gh-pages/tagtree.json","wb") as j:
		import json
		json.dump(tree,j)

if True:
	reduced, v = reduce_data(ldata)
	similar = similarity(reduced.T)
	from scipy.cluster.hierarchy import ward
	clusters = ward(similar)
	w = reduced.sum(axis=0).tolist()[0]
	wordtree = jsontree(clusters,2*clusters.shape[0],v,w,4,np.nan, flatten="hidden")
	with open(path+"gh-pages/wordtree.json","wb") as j:
	#with open(path+"gh-pages/wordtree" + jsontree_tally + ".json","wb") as j:
		import json
		json.dump(wordtree,j)

def jsontree(links, id, names, counts, min_d, min_s,flatten="full"):
	global jsontree_tally
	if id == 2*clusters.shape[0]:
		jsontree_tally = 0
	#if this is a leaf node
	if id <= links.shape[0]:
		jsontree_tally+=1
		return {"name" : names[id], "size" : counts[id]}
	#otherwise, this is a branch node
	node = {}
	#this size variable is actually the number of leaves, not the true size
	left, right, dist, size = tuple(links[int(id-links.shape[0]-1)])
	node["d"] = dist
	#if we prune at this step, flatten the remaining branches
	if dist < min_d or size < min_s: #wait a second...the min size thing does not actually work
		jsontree_tally+=1
		nodes = flatjson(links, names, counts, id)
		#choose method of representing the aggregated nodes
		if flatten=="full":
			#flatten the children but do not prune leaves
			node["name"] = str(int(id))
			node['children'] = [{"name": n['name'], "size": counts[names.index(n['name'])]} for n in nodes]
		elif flatten=="flat":
			#flatten the hierarchy into a string
			node["name"] = str([n['name'] for n in nodes])
			node["size"] = sum([counts[names.index(n['name'])] for n in nodes])
		elif flatten=="lists":
			#flatten the children into a list
			node["name"] = [n['name'] for n in nodes]
			node["size"] = sum([counts[names.index(n['name'])] for n in nodes])
		elif flatten=="hidden":
			#flatten the children into a value, use the largest child as main name
			# better would be to sort by size
			# change title to word wrap?
			best = np.argmax([counts[names.index(n['name'])] for n in nodes])
			node["name"] = nodes[best]['name']+"*"
			node["size"] = sum([counts[names.index(n['name'])] for n in nodes])
			node["value"] = [n['name'] for n in nodes]
		else:
			#dflatten the hierarchy, count the nodes, and drop the name
			node["name"] = str(int(id))
			node["size"] = len(nodes)
	#otherwise, recurse down the branches
	else:
		node["name"] = str(int(id))
		node["children"] = [jsontree(links, int(left), names, counts, min_d, min_s,flatten=flatten),jsontree(links, int(right), names, counts, min_d, min_s,flatten=flatten)]
	if id == 2*clusters.shape[0]:
		print "created a total of " + str(jsontree_tally) + " clusters."
	return node

def flatjson(links, names, counts, id):
	#if this is a leaf node, return the name in a list
	if id <= links.shape[0]:
		return [{'name': names[int(id)], 'size': counts[int(id)]}]
		#return [names[int(id)]]
	#otherwise, return the concatenation of two lists
	left, right, dist, size = tuple(links[int(id-links.shape[0]-1)])
	list1 = flatjson(links, names, counts, left)
	list2 = flatjson(links, names, counts, right)
	#does this always fully flatten the list?
	return list1 + list2





	def word_chisq(	key,
					reference=None,
					data=ndata,
					vocab=vocab,
					n=10,
					minwords=0,
					maxwords=1000,
					stops=[]):
		from sklearn.feature_selection import chi2
		if type(key) == str:
				key = [key]

		if stops == "custom":
			collect = []
			for k in key:
				collect += customstops[k]
			stops = collect

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

			if not np.isnan(chisq[rank]) and not freqs[:,rank]<minwords and not freqs[:,rank]>maxwords and vocab[rank] not in stops:
				values.append((chisq[rank],vocab[rank],p[rank],freqs[:,rank][0,0]))
				i+=1

		return values[0:n]


	y,n,x = "y","n","x"
	def curate(keys, values, stops = customstops, jsonize=False):
		if jsonize:
			with open(path+"data/customstops.json","rb") as f:
				customstops = json.loads(f.read())
		if type(keys) == str:
			keys = [keys]
		j = 0
		for value in values:
			response = input("Use " + str(value) + "? (" + str(j) + " words so far) (y/n/x)")
			if response == "n":
				for key in keys:
					if key not in stops:
						stops[key] = []
					if value[1] not in stops[key]:
						stops[key].append(value[1])
						print "Added " + value[1] + " to the custom stopword list entry for " + key + "."
						j+=1
			elif response == "x":
				break

		if jsonize:
			dumpstops(stops=stops)
		else:
			print "Finished adding stopwords"

	def dumpstops(stops=customstops):
		with open(path+'data/customstops.json', 'w') as outfile:
			import pprint
			json.dump(stops, outfile)
			print "Dumped stopwords to file."

def heatcsv(rows, columns, c):
	cross = False
	if len(rows)==len(columns):
		cross = True
		for n, row in enumerate(rows):
			if rows[n] != columns[n]:
				cross = False
	import matplotlib.pyplot as plt
	from sklearn.feature_selection import chi2
	column_labels = columns
	row_labels = rows
	data = np.zeros((len(rows),len(columns)))
	for y,col in enumerate(columns):
		labels = [(col in row) for row in all_index]
		mat = np.zeros((len(experiences),len(rows)))
		for i,exp in enumerate(experiences):
			for j,row in enumerate(rows):
				if row in all_index[i]:
					mat[i,j] = True
				else:
					mat[i,j] = False
		chisq, p = chi2(mat, labels)
		for x,cs in enumerate(chisq):
			data[x,y] = cs
		print "finished " + col
	if cross==True:
		for y,col in enumerate(columns):
			for x,row in enumerate(rows):
				if x <= y:
					data[x,y] = None
	with open(path + '/gh-pages/' + c + '.csv','wb')  as f:
		import csv
		writer = csv.writer(f)
		writer.writerow(["rowname"] + columns)
		for i in range(len(rows)):
			writer.writerow([rows[i]] + data[i].tolist())

heatcsv(sorted(tag_count.keys()),sorted(subs100.keys()),"heatmap")

	def heatmap(rows, columns):
		cross = False
		if len(rows)==len(columns):
			cross = True
			for n, row in enumerate(rows):
				if rows[n] != columns[n]:
					cross = False
		import matplotlib.pyplot as plt
		from sklearn.feature_selection import chi2
		column_labels = columns
		row_labels = rows
		data = np.zeros((len(rows),len(columns)))
		for y,col in enumerate(columns):
			labels = [(col in row) for row in all_index]
			mat = np.zeros((len(experiences),len(rows)))
			for i,exp in enumerate(experiences):
				for j,row in enumerate(rows):
					if row in all_index[i]:
						mat[i,j] = True
					else:
						mat[i,j] = False
			chisq, p = chi2(mat, labels)
			for x,cs in enumerate(chisq):
				data[x,y] = np.log(cs)
			print "finished " + col
		if cross==True:
			blank = data.min()
			for y,col in enumerate(columns):
				for x,row in enumerate(rows):
					if x <= y:
						data[x,y] = blank
		fig, ax = plt.subplots()
		heatmap = ax.pcolor(data, cmap=plt.cm.Reds)
		# put the major ticks at the middle of each cell
		ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
		ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
		# want a more natural, table-like display
		ax.invert_yaxis()
		if not cross:
			ax.xaxis.tick_top()
		ax.set_xticklabels(column_labels, minor=False, rotation=90)
		ax.set_yticklabels(row_labels, minor=False)
		plt.show()

	heatmap(sorted(tag_count.keys()),sorted(subs100.keys()))
	heatmap(sorted(subs100.keys()),sorted(subs100.keys()))

	def cloud(results):
		words = [row[1] for row in results]
		import webbrowser
		url = "http://infiniteperplexity.github.io/ineffable/wordclouds.html"
		param = ",".join(words)
		webbrowser.open_new(url + "?words=" + param)

	cloud(word_chisq(("datura","brugmansia"),data=ldata, n=50, minwords=25, stops=customstops["datura"]))
	cloud(word_chisq(("lsd"),data=ldata, n=50, minwords=25, maxwords=250, stops="custom"))
	cloud(word_chisq(("meth"),data=ldata, n=50, minwords=25, stops="custom"))

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
