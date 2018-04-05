import numpy as np
import gensim
class Vectorizer:
    def __init__(self, model):
        self.model = model
    '''
    ПРИМЕЧАНИЕ:
        Это на будущее, пока не используется
    Задача:
        Вычисление метрики TF для слова в предложении
    Входные данные:
        word - слово, метрика которого интересует
        sentence - предложение, в окружении которого требуется найти TF
    Результат:
        Значение метрики TF
    '''
    def compute_tf(self, word: str, sentence: list):
        return sum(1.0 for i in sentence if word == i) / len(sentence)
    
    '''
    Задача:
        Поиск среднего по всем векторам вектора
    Входные данные:
        vectorSet - список векторов
    Результат:
        Средний по всем векторам вектор   
    '''
    def calc_mean_vector(self, vector_set: list) -> list:
        return np.mean(vector_set, axis=0)
        
    '''
    Задача:
        Преобразование слов предложения в вектора
    Входные данные:
        words - список слов
    Результат:
        Список векторов    
    '''
    def find_sentence_vector(self, words: list) -> list:
        vectorSet = []
        for word in words:
            try:
                # Загрузка вектора слова из модели
                word_vec = self.model[word]
                vectorSet.append(word_vec)
            except:
                pass
        if vectorSet ==[]:
            return 0
        return self.calc_mean_vector(vectorSet)
    
    
