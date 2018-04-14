from similarity import Similarity
from text_preprocess2 import TextPreprocessor
from vectorizing import Vectorizer
import numpy as np
import pandas as pd
import gensim
import pickle
from progbar import set_progress
pd.options.mode.chained_assignment = None  # default='warn'
def argsortDic(seq):
    return sorted(seq.keys(), key=seq.__getitem__)
def argsortList(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

class Core:
    sim = 0
    text_prep = 0
    tovec = 0
    data = 0
    model = 0
    def __init__(self, model_path: str, data_heh, idf_scores):
        self.load_model(model_path)
        self.sim = Similarity()
        self.text_prep = TextPreprocessor()
        self.tovec = Vectorizer(self.model,idf_scores)
        self.data = self.prepare_data(data_heh)
        
    def prepare_data(self,data):
#        self.data['context_0'] = self.data['context_0'].map(self.tovec.find_sentence_vector)
        print("Векторизация исходного текста...")
        # Векторизация одноконтекстных элементов выборки
        data[0]['sum_words'] =  data[0]['context_0'] 
        data[0]['vec_words'] =  data[0]['context_0'].map(self.tovec.find_sentence_vector)
        data[0]['reply_vec'] =  data[0]['reply'].map(self.tovec.find_sentence_vector)
        
        # Векторизация двухконтекстных элементов выборки
        data[1]['sum_words'] =  data[1]['context_0'] +\
                                data[1]['context_1']
        data[1]['vec_words'] =  data[1]['sum_words'].map(self.tovec.find_sentence_vector)
        data[1]['reply_vec'] =  data[1]['reply'].map(self.tovec.find_sentence_vector)
        
        #Векторизация трехконтекстных элементов выборки
        data[2]['sum_words'] =  data[2]['context_0'] +\
                                data[2]['context_1'] +\
                                data[2]['context_2']
        data[2]['vec_words'] =  data[2]['sum_words'].map(self.tovec.find_sentence_vector)
        data[2]['reply_vec'] =  data[2]['reply'].map(self.tovec.find_sentence_vector)
        
        data = pd.concat([data[0],
                          data[1],
                          data[2]])
        print("Векторизация текста выполнена!")
        return data
        
    def load_model(self, model_path:str)-> bool:
        self.model_exist = True
        print('Загружаю языковую модель...')
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
        print('Загрузка языковой модели прошла успешно!')
        return True
    
        
    
#    def find_answer(self, sentence: str):
#        word_list = self.text_prep.preprocess_sentence(sentence)
#        sentence_vec = self.tovec.find_sentence_vector(word_list)
#        max_cos = -2
#        best_question = []
#        i = 0
#        if np.linalg.norm(sentence_vec) == 0:
#            return []
##        set_progress(0)
#        for row in self.data.itertuples(index=True, name='Pandas'):
#            vec_train = getattr(row, 'vec_words')
#            cos_now = self.sim.CosineSimilarity(vec_train, sentence_vec)
#            if cos_now > max_cos:
#                max_cos = cos_now
#                best_question = []
#            if cos_now == max_cos:
#                best_question.append({'quest':getattr(row, 'sum_words'),'reply':getattr(row,'reply'),'label':getattr(row,'label'),'confidence':getattr(row,'confidence')})
#            i+=1
##        print(max_cos)
#        return best_question
    
    def find_answer(self, sent_vec , nothing:int = 0):
        max_cos = -2
        best_question = []
        i = 0
        if np.linalg.norm(sent_vec) == 0:
            return []
#        set_progress(0)
        for row in self.data.itertuples(index=True, name='Pandas'):
            vec_train = getattr(row, 'vec_words')
            cos_now = self.sim.CosineSimilarity(vec_train, sent_vec)
            if cos_now > max_cos:
                max_cos = cos_now
                best_question = []
            if cos_now == max_cos:
                best_question.append({'quest':getattr(row, 'sum_words'),'reply':getattr(row,'reply'),'reply_vec':getattr(row,'reply_vec'),'label':getattr(row,'label'),'confidence':getattr(row,'confidence')})
            i+=1
#        print(max_cos)
        return best_question
    
    def bad_good_neutral(find_answer_result):
        return {        
                id(d['reply']):d['confidence'] 
                if d['label'] == 'good'
                else (1 - float(d['confidence'])) 
                for d in find_answer_result
                }
        
    '''
    Назначение:
        Сортирует ответные реплики по убыванию полезности
    Входные данные:
        bag - набор строк тестового датафрейма с одним индексом
        find_answer_result - результат выполнения функции self.find_answer для bag
    Результат:
        Список отсортированных reply_id
    '''
    def rerange_replies(self,bag: pd.DataFrame, find_answer_result: list, bgn)-> list:
            # Получаем ответы для лучшего высказывания из обучающей выборки
            if find_answer_result == []:
                replies_conf = {'':0}
                replies_form = {}
            else:
                replies_conf = bgn(find_answer_result)
                replies_form = { 
                        id(d['reply']): d['reply_vec']
                        for d in find_answer_result
                       }
            result = []
            
            conf_matrix = []
            i=0
            
            for row_quest in bag.itertuples(index=True, name='Pandas'):
                conf_matrix.append({'':-2})
                for repl_known in replies_form:
                    # дададададада можно посчитать вектора заранее. мне лениво
                    conf_matrix[i][repl_known]=\
                                        self.sim.CosineSimilarity(
                                                replies_form[repl_known],
                                                getattr(row_quest, 'reply_vec')
                                                )
                            
                i+=1
            
            i=0
            goodness = []
            for quest_reply in conf_matrix:
                # Сортируем по значению похожести входного ответа на ответы из обучающей выборки
                quest_reply_args = max(quest_reply, key=lambda k: quest_reply[k])
                # Выбираем наиболее подходящий и "забираем" его оценку
                goodness.append(replies_conf[quest_reply_args])
                i+=1
            rightRowsNums = argsortList(goodness)    
            result = [bag.iloc[rrn]['reply_id'] for rrn in reversed(rightRowsNums)]
            return result
    '''
    Метод выполняет обработку тестового сета и
    записывает в файл filename результаты тестирования
    (отранжированные ответы)
    '''    
    def do_test(self, test: pd.DataFrame, filename: str = "test_result.txt", bgn = bad_good_neutral) -> None:
        print("Производится решение...")
        idies = sorted(list(set(test['context_id'])))
        lenid = len(idies)
        result_file = open(filename,"w")
        set_progress(0)
        i = 1
        for idd in idies:
            # Выделяем все строчки с нужным id
            bag = test[test['context_id'] == idd]
            # Определяем количество контекстов
            # Производим поиск наилучшего известного примера
            best_quest = self.find_answer(bag['vec_words'].iloc[0])
            # Сортируем имеющиеся ответы
            result = self.rerange_replies(bag, best_quest, bgn)
            # Выводим
            for reply_id in result:
                result_file.write("%i\t%i\n"%(idd,reply_id))
            if i%90==0:
                set_progress(i/lenid)
            i+=1
        result_file.close()
        
        
    
#%%
if __name__=="__main__":
    with open("train_canonized.pickle",'rb') as file:
        data = pickle.load(file)
    with open("idf.pickle",'rb') as file:
        idf_scr = pickle.load(file)
    core = Core("ruwikiruscorpora_upos_skipgram_300_2_2018.vec",data,idf_scr)
#%%
    with open("test_canonized.pickle",'rb') as file:
        testdata = pickle.load(file)
    testdata = core.prepare_data(testdata)
    core.do_test(testdata)
#%%
def checkout(quest):
    print(quest)
    answ = core.find_answer(quest)
    print(' '.join(answ[0]['quest']))
    for line in answ:
        print('================================================')
        print("\tОтвет: ",line['reply'])
        print("\tМетка: ",line['label'])
        print("\tУверенность: ",line['confidence'])

