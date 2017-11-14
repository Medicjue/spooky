#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 00:28:18 2017

baseline Full Program
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


model = RandomForestClassifier(n_jobs=4, random_state=random_seed)
model.fit(one_hot_tokens, labels)

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

test_result.to_csv('tmp/test_full_result.csv', index=False)
