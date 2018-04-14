import numpy as np
import math
import warnings
warnings.filterwarnings("ignore")
class Similarity:

        
    '''
    Задача:
        Поиск косинусного расстояния между двумя предложениями
    Входные данные:
        svec1 - вектор первого предложения
        svec2 - вектор второго предложения
    Результат:
        Косинусное расстояние между двумя векторами предложений
    '''
    def CosineSimilarity(self, svec1, svec2) -> float:
        lng = np.linalg.norm(svec1) * np.linalg.norm(svec2)
        if lng ==0:
            return -1
        cosine_similarity = np.dot(svec1, svec2) / lng            
        return cosine_similarity