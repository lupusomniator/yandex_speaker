
# coding: utf-8

# # YANDEX ML

# In[12]:


import pandas as pd
import pymorphy2
from  TextPreprocessing import TextPreprocessing
import pickle
from  profiler import Profiler


# In[13]:


df = pd.read_csv('train.tsv',
                         names=['context_id', 'context_2', 'context_1', 'context_0', 'reply_id', 'reply',
                                'label', 'confidence'],
                        header=None, sep='\t')


# In[14]:



df.head()


# In[18]:


df_sparce = []
col_name = ['context_0','context_1','context_2']
cont_indexes = [
    pd.isnull(df['context_2']) & pd.isnull(df['context_1']),
    pd.isnull(df['context_2']) & pd.notnull(df['context_1']),
    pd.notnull(df['context_2']) & pd.notnull(df['context_1'])
]
# Один контекст
df_sparce.append(df[cont_indexes[0]])

# Два контекста
df_sparce.append(df[cont_indexes[1]])

# Три контекста
df_sparce.append(df[cont_indexes[2]])


# In[32]:


tp = TextPreprocessing()
col_amount = 0

'''
data_sparced - Список датафреймов
len(data_sparced) = 3
Каждый элемент списка - подвыборка исходного датафрейма
data_sparced[0] - подвыбока, содержащая только один контекст
data_sparced[1] - подвыбока, содержащая два контекста
data_sparced[2] - подвыбока, содержащая три контекста
'''
data_sparced = []
data_sparced_clear = []
with Profiler():
    for df_part in df_sparce:
        data_sparced.append(df[cont_indexes[col_amount]])
        data_sparced_clear.append(df[cont_indexes[col_amount]])
        for i in range(col_amount+1):
            # Текст с тегами
            prep_col = df_part[col_name[i]].map(tp.preprocess_sentense_tagged)
            data_sparced[col_amount][col_name[i]] = prep_col
            # Без тегов
            prep_col_clear = df_part[col_name[i]].map(tp.preprocess_sentence)
            data_sparced[col_amount][col_name[i]] = prep_col_clear
            
        col_amount+=1

# In[33]:


with Profiler():
    with open("canonized_clear.pickle",'wb') as file:
        pickle.dump(data_sparced, file)
        
with Profiler():
    with open("canonized_tagged.pickle",'wb') as file:
        pickle.dump(data_sparced, file)
'''
Пример загрузки файла из пикла
        
with open("kfg",'rb') as file:
    asd = pickle.load(file)
'''

