
# coding: utf-8

# In[36]:

import codecs
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# In[39]:


import pymorphy2
from string import punctuation
from nltk.corpus import stopwords


# In[72]:


class TextPreprocessor:
    
    morph = pymorphy2.MorphAnalyzer()
    stop_words = stopwords.words('russian')
    
    def __init__(self):
        print()
        
    '''
    Перевести тег слова из формата Pymorphy2 в общепринятый формат
    '''
    def translate_tags(self, tag: str) -> str:
        grammars = {'NOUN': '_NOUN', 'ADJF': '_ADJ', 'ADJS': '_ADJ', 'COMP': '_ADV', 'VERB': '_VERB', 'INFN': '_VERB',
                    'GRND': '_VERB', 'PRTF': '_ADJ', 'PRTS': '_ADJ', 'NUMR': '_NUM','NUMB': '_NUM', 'ADVB': '_ADV', 'NPRO': '_PRON',
                    'PRED': '_ADV', 'PREP': '_ADP', 'CONJ': '_CCONJ', 'PRCL': '_PART', 'INTJ': '_INTJ', 'PNCT' : '_PUNCT'}
        if tag in grammars:
            return grammars[tag]
        else:
            return ''


    
    def preprocess_sentence(self, sentence: str)->list:
        normalized = []
        # Удаляем всю пунктуацию в предложении и
        # объеденяем слова в предложении в список
        sentence_list = nltk.word_tokenize(sentence.lower().replace('ё','е'))
        lastone = ''
        # Для каждого слова в предложении
        for word in sentence_list:
            # Проверка на отсутствие слова в списке стоп-слов
            # Находим все возможные варианты разбора слова    
            forms = self.morph.parse(word)
            try:
                # Выбираем наиболее вероятный вариант
                form = max(forms, key=lambda x: x.score)
            except Exception:
                # Если разбор слова не удался, просто оставляем его как есть
                form = forms[0]
            strtag = str(form.tag)
            if form.score >= 0.4:
                if any(['Name' in strtag,'Surn' in strtag,'Patr' in strtag]):
                    if lastone != 'персонаж_NOUN':
                        lastone = 'персонаж_NOUN'
                        normalized.append('персонаж_NOUN')
                    continue
                if ('Geox' in strtag):
                    if lastone != 'локация_NOUN':
                        lastone = 'локация_NOUN'
                        normalized.append('локация_NOUN')
                    continue
                if any(['Orgn' in strtag, 'Trad' in strtag]):
                    if lastone != 'организация_NOUN':
                        lastone = 'организация_NOUN'
                        normalized.append('организация_NOUN')
                    continue
                if ('NUMR' in strtag or 'NUMB' in strtag):
                    if lastone != 'число_NUM':
                        lastone = 'число_NUM'
                        normalized.append('число_NUM')
                    continue
            if 'PRCL' in strtag:
                continue
            to_app = form.normal_form.replace('ё', 'е') + self.translate_tags(strtag[0:4])
            normalized.append(to_app)
            lastone = to_app
        return normalized
        
        
        
    '''
    Выполнить предобработку корпуса предложений
    '''
    def preprocess_text(self, sentence_list: list) -> list:
        result = []
        for sentence in sentence_list:
            result.append(self.preprocess_sentence(sentence))
        return result
    

    def list_to_text(self, sentence_list: list) -> str:
        return ' '.join(word for word in sentence_list)

    
if __name__ == "__main__":
    p = TextPreprocessor()
    output = codecs.open("Processed_sent4.txt","w",'utf-8')
    with open("OpenSubtitles2016.en-ru.ru.utf-8", "r",encoding='utf-8') as f:
        for i, l in enumerate(f):
            new_cent = p.list_to_text(p.preprocess_sentence(l))
            try:
                output.write(new_cent + "\n")
            except:
                pass
            if (i%1000 == 0):
                print(i)
	
	
#    file = open('train_txt.txt', 'r')
#    file_out = open('shit.txt','w')
#    for line in file:
#        prep_line = p.preprocess_sentence(line)
#        for word in prep_line:
#            if not '_' in word and (word!='neutral' and word!='good' and word!='bad'):
#                file_out.write(word+'\n')
#    

