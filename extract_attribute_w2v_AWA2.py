#!/usr/bin/env python3

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import pdb
import pandas as pd
import numpy as np
import gensim.downloader as api
import scipy.io as sio
import pickle
from gensim.models import KeyedVectors
import csv
#%%
print('Loading pretrain w2v modeling')
model_path = '/pretrained_models/GoogleNews-vectors-negative300.bin.gz'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
dim_w2v = 300
print('Done loading modeling')
#%%
replace_word = [('newworld','new world'),('oldworld','old world'),('nestspot','nest spot'),('toughskin','tough skin'),
                ('longleg','long leg'),('chewteeth','chew teeth'),('meatteeth','meat teeth'),('strainteeth','strain teeth'),
                ('quadrapedal','four feet'),('longneck','long neck'),('buckteeth','buck teeth'),('flys','fly'),('bipedal','two feet')]
dataset = 'AwA2'
#%%
with open('/datasets/AWA2/Animals_with_Attributes2/predicates.txt', 'r') as file:
    lines = file.readlines()
data = []
for line in lines:
    fields = line.strip().split('\t')  # 根据实际情况选择适当的分隔符
    data.append(fields)
with open('//tools/AWA_att.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

df=pd.read_csv('/tools/AWA_att.csv')
des = df['des']
print(des)

#%% replace out of dictionary words
for pair in replace_word:
    for idx,s in enumerate(des):
        des[idx]=s.replace(pair[0],pair[1])
print('Done replace OOD words')

#%%
counter_err = 0
all_w2v = []
for s in des:
    print(s)
    words = s.split(' ')
    if words[-1] == '':     #remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    for w in words:
        try:
            w2v += model[w]
        except Exception as e:
            print(e)
            counter_err += 1
    all_w2v.append(w2v[np.newaxis,:])
print('counter_err ',counter_err)
#%%
all_w2v=np.concatenate(all_w2v,axis=0)
# pdb.set_trace()
#%%
with open('datasets/Attribute/w2v/{}_attribute.pkl'.format(dataset),'wb') as f:
    pickle.dump(all_w2v,f)    
