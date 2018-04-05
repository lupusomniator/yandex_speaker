import pickle
import gensim
import pymorphy2
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence, Doc2Vec

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

#%%

#model = gensim.models.Word2Vec(sentences, min_count=5, window = 5)
#model.wv.most_similar_cosmul(positive=['город'])

model = gensim.models.Word2Vec(alpha=0.025, min_alpha=0.025)
print("GENISMED")
model.build_vocab(sentences)
print("VOCAB")
for epoch in range(10):
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
print("TRAINED")
#%%
model = gensim.models.KeyedVectors.load_word2vec_format("C:\\Users\\User\\OneDrive\\MLTrack\\yandex_speaker\\ruscorpora_upos_skipgram_300_5_2018.vec", binary=False)
#%%
print(model.wv.most_similar(positive=['любить_VERB','собака_NOUN']))
