
# coding: utf-8

# In[36]:


import nltk
nltk.download('stopwords')


# In[39]:


import pymorphy2
from string import punctuation
from nltk.corpus import stopwords


# In[72]:


class TextPreprocessing:
    
    morph = pymorphy2.MorphAnalyzer()
    stop_words = stopwords.words('russian')
    
    def __init__(self):
        print()
        
    '''
    Перевести тег слова из формата Pymorphy2 в общепринятый формат
    '''
    def translate_tags(self, tag: str) -> str:
        grammars = {'NOUN': '_NOUN', 'ADJF': '_ADJ', 'ADJS': '_ADJ', 'COMP': '_IGN', 'VERB': '_VERB', 'INFN': '_VERB',
                    'GRND': '_VERB', 'PRTF': '_VERB', 'PRTS': '_VERB', 'NUMR': '_NUM', 'ADVB': '_ADV', 'NPRO': '_PRON',
                    'PRED': '_ADV', 'PREP': '_ADP', 'CONJ': '_CCONJ', 'PRCL': '_PART', 'INTJ': '_INTJ'}
        if tag in grammars:
            return grammars[tag]
        else:
            return ''
        
    '''
    Удалить всю пунктуацию в предложении
    '''
    def remove_punctuation(self, s: str) -> str:
        return ''.join(c if c not in punctuation else ' ' for c in s)
    
    '''
    Выполнить предобработку одного предложения
    '''
    def preprocess_sentence(self, sentence: str)-> list: 
        normalized = []
        # Удаляем всю пунктуацию в предложении и
        # объеденяем слова в предложении в список
        sentence_list = self.remove_punctuation(sentence).split()
        
        # Для каждого слова в предложении
        for word in sentence_list:
            # Проверка на отсутствие слова в списке стоп-слов
            if word in self.stop_words:
                continue
            # Находим все возможные варианты разбора слова    
            forms = self.morph.parse(word)
            try:
                # Выбираем наиболее вероятный вариант
                form = max(forms, key=lambda x: x.score)
            except Exception:
                # Если разбор слова не удался, просто оставляем его как есть
                form = forms[0]
            # Если не удалось определить тип слова или нормальная форма слова находится в стоп-словах       
            if not ('Name' in form.tag or 'UNKN' in form.tag or 'LATN' in form.tag or form.normal_form in self.stop_words):
                # RusVectories требует отсутствия букв ё
                #if form.normal_form.replace('ё', 'е') + self.translate_tags(form.tag.POS) in word_vec:
                normalized.append(form.normal_form.replace('ё', 'е'))
        # else:
        #     normalized.append(word)
        return normalized
    
    def preprocess_sentense_tagged(self, sentence: str)->list:
        normalized = []
        # Удаляем всю пунктуацию в предложении и
        # объеденяем слова в предложении в список
        sentence_list = self.remove_punctuation(sentence).split()
        
        # Для каждого слова в предложении
        for word in sentence_list:
            # Проверка на отсутствие слова в списке стоп-слов
            if word in self.stop_words:
                continue
            # Находим все возможные варианты разбора слова    
            forms = self.morph.parse(word)
            try:
                # Выбираем наиболее вероятный вариант
                form = max(forms, key=lambda x: x.score)
            except Exception:
                # Если разбор слова не удался, просто оставляем его как есть
                form = forms[0]
            # Если не удалось определить тип слова или нормальная форма слова находится в стоп-словах       
            if not ('Name' in form.tag or 'UNKN' in form.tag or 'LATN' in form.tag or form.normal_form in self.stop_words):
                # RusVectories требует отсутствия букв ё
                #if form.normal_form.replace('ё', 'е') + self.translate_tags(form.tag.POS) in word_vec:
                normalized.append(form.normal_form.replace('ё', 'е') + self.translate_tags(form.tag.POS))
        # else:
        #     normalized.append(word)
        return normalized
        
        
        
    '''
    Выполнить предобработку корпуса предложений
    '''
    def preprocess_text(self, sentence_list: list, tagged: bool = True) -> list:
        result = []
        if tagged:
            for sentence in sentence_list:
                result.append(preprocess_sentence_tagged(sentence))
        else:    
            for sentence in sentence_list:
                result.append(preprocess_sentence(sentence))
        return result
    
    '''
    Выполнить предобработку корпуса предложений
    '''
    def list_to_text(self, sentence_list: list) -> str:
        return ' '.join(word for word in sentence_list)

    
if __name__ == "__main__":
    p = TextPreprocessing()
    print(p.preprocess_sentence("!@#$%^&*().,/<>\{}[]"))
    

