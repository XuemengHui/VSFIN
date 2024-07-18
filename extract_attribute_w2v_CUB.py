# -*- coding: utf-8 -*-

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
replace_word = [('spatulate','broad'),('upperparts','upper parts'),('grey','gray'),('eyeline', 'eye line'), ('eyering', 'eye ring')]
#%%
with open('datasets/CUB/attributes.txt', 'r') as file:
    lines = file.readlines()
data = []
for line in lines:
    fields = line.strip().split('\t')  # 根据实际情况选择适当的分隔符
    data.append(fields)
with open('/tools/CUB_att.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

df =pd.read_csv('/tools/CUB_att.csv')
des = df['des']
print(des)
#%% filter
new_des = [' '.join(i.split('_')) for i in des]
new_des = [' '.join(i.split('-')) for i in new_des]
new_des = [' '.join(i.split('::')) for i in new_des]
new_des = [i.split('(')[0] for i in new_des]
new_des_1 = [i[6:] for i in new_des[:9]]
new_des_2 = [i[7:] for i in new_des[9:99]]
new_des_3 = [i[8:] for i in new_des[99:]]
new_des = new_des_1 + new_des_2 + new_des_3
#%% replace out of dictionary words
for pair in replace_word:
    for idx,s in enumerate(new_des):
        new_des[idx]=s.replace(pair[0],pair[1])
print('Done replace OOD words')
#%%
all_w2v = []
for s in new_des:
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
    all_w2v.append(w2v[np.newaxis,:])
#%%
all_w2v=np.concatenate(all_w2v,axis=0)
# pdb.set_trace()
#%%
with open('datasets/Attribute/w2v/CUB_attribute.pkl','wb') as f:
    pickle.dump(all_w2v,f)