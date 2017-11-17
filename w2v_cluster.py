#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 22:31:53 2017

Word2Vec Clustering
"""
import numpy as np
import gc
import pandas as pd
import pickle

from sklearn.cluster import KMeans

import datetime

random_seed = 23

words = []
words_vec = []

print('Read Data')
start = datetime.datetime.now()
f = open('data/wiki.en.vec', 'r', encoding='utf8')
print(f.readline())
index = 0
for line in f:
    wv = line.split(' ')
    word = wv[0]
    vec = np.asarray(wv[1:len(wv)-1], dtype=float)
    words.append(word)
    words_vec.append(vec)
    index += 1
f.close()
end = datetime.datetime.now()
gc.collect()

#print('Output File')
#f = open('data/words.np', 'wb')
#pickle.dump(words, f)
#f.close()
#
#f = open('data/wv.np', 'wb')
#pickle.dump(words_vec, f)
#f.close()

print('Read Data Done, time consume: {}'.format(end-start))
start = datetime.datetime.now()
kmeans = KMeans(n_clusters=20, random_state=random_seed, n_jobs=-1).fit(words_vec)
end = datetime.datetime.now()
print('KMeans Done, time consume: {}'.format(end-start))

result = pd.DataFrame()
result['word'] = pd.Series(words)
result['cluster'] = pd.Series(kmeans.labels_)
result.to_csv('tmp/cluster_20_result.csv', index=False)