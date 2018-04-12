import numpy as np
import pandas as pd
import pickle
from text_preprocess2 import TextPreprocessor
tp = TextPreprocessor()
df = pd.read_csv('freqrnc2012.csv', sep='\t')
words = df['Lemma'].values
words_docs = df['Doc'].values
print(words)
i = 0
for i in range(len(words)):
    words[i] = tp.preprocess_sentence(words[i])
    i+=1
#%%
words = list(map(lambda x:x[0],words))
#%%
dic_words_idf = {}
i = 0
for i in range(len(words)):
    dic_words_idf[words[i]] = np.log(38369/words_docs[i])
with open('idf.pickle','wb') as file:
    pickle.dump(dic_words_idf,file)