import pickle
import itertools
import collections

with open("canonized_clear.pickle",'rb') as file:
    asd = pickle.load(file)
#%%
print("LOADED")
sentences = list(asd[0]['context_0'])
sentences.extend(asd[1]['context_0'])
sentences.extend(asd[1]['context_1'])
sentences.extend(asd[2]['context_0'])
sentences.extend(asd[2]['context_1'])
sentences.extend(asd[2]['context_2'])

wordsDictionary = collections.Counter()

words = set(tuple(x) for x in sentences)
words = [ list(x) for x in b_set ]

words = list(set(list(itertools.chain(*words))))

for word in words:
    for sent in sentences:
        if word in sent:
            wordsDictionary[word] += 1


#%%
TFIDF = {}
for word in words:
     TFIDF[word]=np.log(float(len(sentences))/wordsDictionary[word])
     
with open("idf",'wb') as file:
    pickle.dump(TFIDF, file)    
 
#print (wordsDictionary)