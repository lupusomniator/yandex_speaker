
# coding: utf-8

# # YANDEX ML

# In[1]:


import pandas as pd
import pymorphy2
from  TextPreprocessing import TextPreprocessing
import pickle
from  profiler import Profiler

# In[2]:


df = pd.read_csv('train.tsv',
                         names=['context_id', 'context_2', 'context_1', 'context_0', 'reply_id', 'reply',
                                'label', 'confidence'],
                        header=None, sep='\t')


# In[3]:



df.head()


# In[6]:


df_sparce = []
col_name = ['context_0','context_1','context_2']
# Один контекст
df_sparce.append(df[pd.isnull(df['context_2']) & pd.isnull(df['context_1'])])

# Два контекста
df_sparce.append(df[pd.isnull(df['context_2']) & pd.notnull(df['context_1'])])

# Три контекста
df_sparce.append(df[pd.notnull(df['context_2']) & pd.notnull(df['context_1'])])


# In[10]:


tp = TextPreprocessing()
col_amount = 1
df_canon = []
with Profiler():
    for df_part in df_sparce:
        df_canon.append([])
        for i in range(col_amount):
            df_canon[col_amount-1].append((df_part[col_name[i]].map(tp.preprocess_sentence)).map(tp.list_to_text))
        col_amount+=1


# In[11]:
with Profiler():
    with open("canonized",'wb') as file:
        pickle.dump(df_canon, file)
