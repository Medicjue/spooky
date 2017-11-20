#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 22:31:53 2017

Word2Vec Clustering
"""
import numpy as np
import gc
import pandas as pd
#import pickle

from sklearn.cluster import KMeans

import datetime

random_seed = 23

k_range = [80]

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

words_vec = np.asarray(words_vec)
words_vec = np.append(words_vec, [words_vec.mean(axis=0)], axis=0)
words.append('MMMMMMMM')

#print('Output File')
#f = open('data/words.np', 'wb')
#pickle.dump(words, f)
#f.close()

#f = open('data/wv.np', 'wb')
#pickle.dump(words_vec, f, protocol=4)
#f.close()

print('Read Data Done, time consume: {}'.format(end-start))
for k in k_range:
    start = datetime.datetime.now()
    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_jobs=6).fit(words_vec)
    end = datetime.datetime.now()
    print('KMeans Done, time consume: {}'.format(end-start))
    
    result = pd.DataFrame()
    result['word'] = pd.Series(words)
    result['cluster'] = pd.Series(kmeans.labels_)
    result.to_csv('tmp/cluster_{}_result.csv'.format(k), index=False)
    del(result)
    gc.collect()