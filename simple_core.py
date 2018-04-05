from similarity import Similarity
from text_preprop import TextPreprocessor
from vectorizing import Vectorizer
import numpy as np
import pandas as pd
import gensim
import pickle
from progbar import set_progress
class Core:
    sim = 0
    text_prep = 0
    tovec = 0
    data = 0
    model = 0
    def __init__(self, model_path: str, data: pd.DataFrame):
        self.load_model(model_path)
        self.sim = Similarity()
        self.text_prep = TextPreprocessor()
        self.tovec = Vectorizer(self.model)
        self.data = data
        self.prepare_data()
        
    def prepare_data(self):
#        self.data['context_0'] = self.data['context_0'].map(self.tovec.find_sentence_vector)
        self.data_vectors = self.data['context_0'].map(self.tovec.find_sentence_vector)
        
    def load_model(self, model_path:str)-> bool:
        self.model_exist = True
        print('Загружаю модель...')
        try:
            if model_path.endswith('.bin'):
                self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
            elif model_path.endswith('.vec'):
                self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
            else:
                self.model = gensim.models.Word2Vec.load(model_path)
        except Exception:
            self.model_exist = False
            print('Загрузка модели не удалась...')
            return False
        self.model.init_sims(replace=True)
        print('Загрузка модель прошла успешно...')
        return True
    
        
    
    def find_answer(self, sentence: str):
        word_list = self.text_prep.preprocess_sentense_tagged(sentence)
        sentence_vec = self.tovec.find_sentence_vector(word_list)
        max_cos = -1
        best_question = []
        i = 0
        le = self.data_vectors.size
#        set_progress(0)
        for line in self.data_vectors:
            cos_now = self.sim.CosineSimilarity(line,sentence_vec)
            if cos_now> max_cos:
                max_cos = cos_now
                best_question = []
            if cos_now == max_cos:
                best_question.append([self.data.values[i][3],self.data.values[i][5],self.data.values[i][7]])
            i+=1
#            set_progress(i/le)
        return best_question
    
#%%
with open("canonized_tagged.pickle",'rb') as file:
    data = pickle.load(file)
core = Core("ruwikiruscorpora_upos_skipgram_300_2_2018.vec",data[0])
#%%
for line in core.find_answer("Что мне купить в этом магазине?"):
    print(line)
#%%
for line in core.find_answer("Почему бы тебе не развестись?"):
    print(line)
#%%
for line in core.find_answer("Он не дышит, только если через соус в спагетти"):
    print(line)
#%%
for line in core.find_answer("Люк я твой отец"):
    print(line)
