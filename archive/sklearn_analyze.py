import pickle, numpy as np
path = 'C:/Users/Glenn/Documents/GitHub/mining-erowid/'



vocab = np.array(vectorizer.get_feature_names())

#create lists of all tags and substances
def iter_flatten(iterable):
  it = iter(iterable)
  for e in it:
    if isinstance(e, (list, tuple)):
      for f in iter_flatten(e):
        yield f
    else:
      yield e
  
tags = set([tag for tag in iter_flatten(tag_index)])
substances = set([substance for substance in iter_flatten(substance_index)])


from sklearn.feature_selection import chi2
tags_distinct = {}
for tag in tags:
	print("Calculating " + tag)
	labels = [(tag in t) for t in tag_index]
	chisq, p = chi2(data, labels)
	ranking = np.argsort(chisq)[::-1]
	values = []
	for rank in ranking
	
	values = []
	for rank in ranking:
		values.append((chisq[rank],vocab[rank],p[rank]))
	tags_distinct[tag] = values

substance_distinct = {}
for substance in substances:
	print("Calculating " + substance)
	labels = [(substance in s) for s in substance_index]
	chisq, p = chi2(data, labels)
	ranking = np.argsort(chisq)[::-1]
	values = []
	for rank in ranking:
		values.append((chisq[rank],vocab[rank],p[rank]))
	substance_distinct[substance] = values
	
	
	X = 
