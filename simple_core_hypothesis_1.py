from similarity import Similarity
from text_preprocess2 import TextPreprocessor
from vectorizing import Vectorizer
import numpy as np
import pandas as pd
import gensim
import pickle
from sklearn.cross_validation import KFold
from progbar import set_progress
from simple_core import Core
pd.options.mode.chained_assignment = None  # default='warn'
def argsortDic(seq):
    return sorted(seq.keys(), key=seq.__getitem__)
def argsortList(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

class Hipothesis1(Core):
    alpha = 0
    beta = 0
    counter = 0
    def bad_good_neutral_hyp1(self,find_answer_result):
        return {        
                id(d['reply']): 1 - float(d['confidence'])
                if d['label'] == 'bad'
                else float(d['confidence']) + self.alpha
                if d['label'] == 'neutral'
                else float(d['confidence']) + self.alpha + self.beta
                for d in find_answer_result
                }
    def get_score(self, sorted_repl, alldata):
        idies = np.array(sorted(list(set(sorted_repl['context_id']))))
        n = len(idies)
        nDCG = 0
        logdic={2:np.log(2),
                3:np.log(3),
                4:np.log(4),
                5:np.log(5),
                6:np.log(6),
                7:np.log(7)
                }
        translate = {'good':2,'neutral':1,'bad':0}
        for idd in idies:
            bag = sorted_repl[sorted_repl['context_id'] == idd]
            DGCp = 0
            i = 1
            rows_data = alldata[alldata['context_id'] == idd]
            for row_quest in bag.itertuples(index=True, name='Pandas'):
                row_data = rows_data[rows_data['reply_id'] == getattr(row_quest, 'reply_id')]
                DGCp += translate[row_data['label'].values[0]] / logdic[i+1]
                i+=1
            goodm = rows_data[rows_data['label'] == 'good'].shape[0]
            neutralm = rows_data[rows_data['label'] == 'neutral'].shape[0]
            IDCGp = 0
            for i in range(1,goodm+1):
                IDCGp += 2/logdic[i+1]
            for i in range(goodm + 1,goodm + neutralm + 1):
                IDCGp += 1/logdic[i+1]
            if IDCGp == 0: 
                continue
            nDCG += DGCp/IDCGp
        return nDCG/n * 100000
                
    def checkHyperParam(self, n_folds: int,alpha, beta, random_state: int = 42)-> dict:
        self.alpha = alpha
        self.beta = beta
        print("\nИсследуемые параметры:\nalpha %f\nbeta %f"%(alpha,beta))
        idies = np.array(sorted(list(set(self.data['context_id']))))
        n = len(idies)
        kf = KFold (n = n,
                   n_folds = n_folds,
                   shuffle = True,
                   random_state = random_state)
        all_data = self.data
        score_table = []
        i=  1
        for train_index, test_index in kf:
            print("\n%i-ый фолд"%i)
            self.data = all_data[all_data['context_id'].isin(idies[train_index])]
            test = all_data[all_data['context_id'].isin(idies[test_index])]['context_id','reply_id','reply']
            self.alpha = alpha
            self.beta = beta
            self.do_test(test, "cross_val.txt", bgn = self.bad_good_neutral_hyp1)
            sorted_repl = pd.read_csv('cross_val.txt',
                                      names=['context_id',  'reply_id'],
                                      header=None, sep='\t')
            
            score_table.append(self.get_score(sorted_repl,all_data))
            i+=1
        self.data = all_data
        return np.mean(score_table)
#%%
with open("idf.pickle",'rb') as file:
    idf_scr = pickle.load(file)
with open("train_canonized.pickle",'rb') as file:
    data = pickle.load(file)
core = Hipothesis1("ruwikiruscorpora_upos_skipgram_300_2_2018.vec",data,idf_scr)
file = open("hyperparams.txt","w")
file.write("alpha\tbeta\tscore\n")
file.close()
for alpha in np.linspace(0,1,4):
    for beta in np.linspace(0,1,4):
            score = core.checkHyperParam(4,alpha,beta)
            file = open("hyperparams.txt","a")
            file.write("%f\t%f\t%f\n"%(alpha,beta,score))
            file.close()

