
# coding: utf-8

# # YANDEX ML

# In[12]:


import pandas as pd
import pymorphy2
from  text_preprocess2 import TextPreprocessor
import pickle
from  profiler import Profiler

# In[13]:


df = pd.read_csv('public.tsv', names=['context_id',
                                'context_2', 'context_1', 'context_0',
                                'reply_id', 'reply'],
                        header=None, sep='\t')

#%%
df_p = df[['context_id','reply_id','reply']]
nan_idies = set(df_p[df_p.isnull().any(axis=1)]['context_id'])
nan_indexes = df_p[df_p.isnull().any(axis=1)].index
print(df.iloc[nan_indexes])







