#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:29:31 2017

baseline program
"""

import pandas as pd
import jieba
import numpy as np
import gc
import pickle

from sklearn.ensemble import RandomForestClassifier

random_seed = 23

train_data = pd.read_csv('data/train.csv')

vocabs = set()
raw_tokens = []
labels = []

for index, row in train_data.iterrows():
    text = row[1].lower()
    label = row[2]
    tokens = set(jieba.lcut(text))
    raw_tokens.append(tokens)
    vocabs |= tokens
    labels.append(label)
    
    
vocabs = list(vocabs)
vocab_size = len(vocabs)
one_hot_tokens = []

for tokens in raw_tokens:
    one_hot_token = np.zeros(vocab_size+1, dtype=bool)
    for token in tokens:
        try:
            one_hot_token[vocabs.index(token)] = True
        except:
            ## Mark as others
            one_hot_token[vocab_size] = True
    one_hot_tokens.append(one_hot_token)
    
gc.collect()

ttl_size = len(one_hot_tokens)
train_size = int(ttl_size * 0.7)

train_x = one_hot_tokens[0:train_size]
train_y = labels[0:train_size]

test_x = one_hot_tokens[train_size:]
test_y = labels[train_size:]

model = RandomForestClassifier(n_jobs=4, random_state=random_seed)
model.fit(train_x, train_y)

predict_y = model.predict(test_x)
predict_y_prob = model.predict_proba(test_x)

result = pd.DataFrame()
result['actual'] = pd.Series(test_y)
result['predict'] = pd.Series(predict_y)

prob = pd.DataFrame(predict_y_prob, columns=['EAP','HPL','MWS'])
result['EAP_prob'] = prob['EAP']
result['HPL_prob'] = prob['HPL']
result['MWS_prob'] = prob['MWS']

result.to_csv('tmp/result.csv', index=False)

del(one_hot_tokens)
del(labels)
del(train_x)
del(train_y)
del(test_x)
del(test_y)
gc.collect()

f = open('mdl/rf.model', 'wb')
pickle.dump(model, f)
f.close()

test_data = pd.read_csv('data/test.csv')

ids = []
test_one_hot_tokens = []
for index, row in test_data.iterrows():
    ids.append(row[0])
    text = row[1].lower()
    tokens = set(jieba.lcut(text))
    one_hot_token = np.zeros(vocab_size+1, dtype=bool)
    for token in tokens:
        try:
            one_hot_token[vocabs.index(token)] = True
        except:
            ## Mark as others
            one_hot_token[vocab_size] = True
    test_one_hot_tokens.append(one_hot_token)
    
test_result = pd.DataFrame()
test_result['id'] = pd.Series(ids)

test_data_prob = model.predict_proba(test_one_hot_tokens)
test_data_prob = pd.DataFrame(test_data_prob, columns=['EAP','HPL','MWS'])

test_result['EAP'] = test_data_prob['EAP']
test_result['HPL'] = test_data_prob['HPL']
test_result['MWS'] = test_data_prob['MWS']

test_result.to_csv('tmp/test_result.csv', index=False)
