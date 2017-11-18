#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 23:33:49 2017

Convert data to histogram vector
"""

import pandas as pd
import gc
import pickle
import jieba
import numpy as np

random_seed = 23

cluster_40 = pd.read_csv('tmp/cluster_40_result.csv')

cluster_map = dict()
for index, row in cluster_40.iterrows():
    cluster_map[row[0]] = row[1]
    
del(cluster_40)
gc.collect()

f = open('data/cluster_40_map.dict', 'wb')
pickle.dump(cluster_map, f)
f.close()
del(f)
gc.collect()


train_data = pd.read_csv('data/train.csv')

raw_tokens = []
#labels = []

for index, row in train_data.iterrows():
    text = row[1].lower()
#    label = row[2]
    tokens = set(jieba.lcut(text))
    raw_tokens.append(tokens)
#    labels.append(label)
    
histograms = []
for raw_token in raw_tokens:
    hist = np.zeros(40, dtype=int)
    for token in raw_token:
        try:
            index = cluster_map[token]
        except:
            index = cluster_map['MMMMMMMM']
        hist[index] = hist[index] + 1
    hist = hist/sum(hist)
    histograms.append(hist)
    
histograms = np.asarray(histograms, dtype=float)

f = open('data/histograms_k40.np', 'wb')
pickle.dump(histograms, f)
f.close()
del(f)
gc.collect()

#labels = np.asarray(labels, dtype=str)
#f = open('data/labels.np', 'wb')
#pickle.dump(labels, f)
#f.close()
#del(f)
#gc.collect()


test_data = pd.read_csv('data/test.csv')

raw_tokens = []
#test_ids = []

for index, row in test_data.iterrows():
#    test_id = row[0]
#    test_ids.append(test_id)
    text = row[1].lower()
    tokens = set(jieba.lcut(text))
    raw_tokens.append(tokens)
    
test_histograms = []
for raw_token in raw_tokens:
    hist = np.zeros(40, dtype=int)
    for token in raw_token:
        try:
            index = cluster_map[token]
        except:
            index = cluster_map['MMMMMMMM']
        hist[index] = hist[index] + 1
    hist = hist/sum(hist)
    test_histograms.append(hist)
    
test_histograms = np.asarray(test_histograms, dtype=float)
f = open('data/test_histograms_k40.np', 'wb')
pickle.dump(test_histograms, f)
f.close()
del(f)
gc.collect()

#test_ids = np.asarray(test_ids, dtype=str)
#f = open('data/test_ids.np', 'wb')
#pickle.dump(test_ids, f)
#f.close()
#del(f)
#gc.collect()