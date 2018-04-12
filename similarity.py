import numpy as np
import math

class Similarity:
    def __init__(self):
        print('hi')
        
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
        cosine_similarity = np.dot(svec1, svec2) / (
                    np.linalg.norm(svec1) * np.linalg.norm(svec2))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity = -1
        except:
            cosine_similarity = 0
            
        return cosine_similarity