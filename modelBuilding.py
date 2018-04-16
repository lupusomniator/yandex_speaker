import pickle
import gensim
import pymorphy2
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#%%
modelLen = 7000000

sentences = []
with open("Processed_sent3.txt",'r', encoding = 'utf-8') as file:
    i = 0
    while (i < modelLen):
        if (not i % 10000):
            print("Loaded " + str(i) + " sentences")
        sentences.append(file.readline().split())
        i+=1
print("Loaded " + str(modelLen) + " sentences")
#print("All sentences loaded. Mapping over list")
#
#sentences = list(map(lambda x: x.split(),sentences))
print ('Data loaded')
#%%

model = gensim.models.Word2Vec(alpha=0.025, min_alpha=0.025, min_count=1)
print("Model created")

model.build_vocab(sentences)

model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
print("Vocab built")
for epoch in range(10):
    print ("Training, iteration " + str(epoch))
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
print("TRAINED")

#%%
with open("model_on_subtitles_v2_" + str(modelLen) + ".pickle","wb") as pcl:
    pickle.dump(model, pcl)


#%%
print(model.wv.most_similar(positive=['ребенок_NOUN']))
