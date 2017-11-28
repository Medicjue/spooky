#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:32:25 2017

SVM (Linear) + NLTK preprocessing
"""

import pandas as pd
import jieba
import numpy as np
import gc
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

random_seed = 23

stop = set(stopwords.words('english'))

stemmer = SnowballStemmer('english')

punctuation = set([',', '.', ' ', '"', '\'', '?', '-', '+', '=', '!'])


def work_prep(sentence):
#    words = word_tokenize(text)
#    words_with_tag = pos_tag(words)
#    tokens = []
#    for word_with_tag in words_with_tag:
#        if 'JJ' in word_with_tag[1]:
#            tokens.append(word_with_tag[0])
#    tokens = set(tokens)
    tokens = set(word_tokenize(sentence))
    tokens = tokens - stop
    tokens = tokens - punctuation
    new_tokens = []
    for token in tokens:
        new_tokens.append(recursive_stem(token))
    return set(new_tokens)
    
    
def recursive_stem(token):
    new_token = stemmer.stem(token)
    if token == new_token:
        return token
    else:
        return recursive_stem(new_token)

train_data = pd.read_csv('data/train.csv')

vocabs = set()
raw_tokens = []
labels = []

for index, row in train_data.iterrows():
    text = row[1].lower()
    label = row[2]
    tokens = work_prep(text)
    raw_tokens.append(tokens)
    vocabs |= tokens
    labels.append(label)

print('collect tokens done')
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

print('convert 2 one-hot done')

#model = RandomForestClassifier(n_jobs=4, random_state=random_seed)
model = SVC(kernel="linear", C=1000, probability=True, random_state=random_seed)
model.fit(one_hot_tokens, labels)
print('model training done')
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
    
gc.collect()
print('convert test data 2 one-hot done')

test_result = pd.DataFrame()
test_result['id'] = pd.Series(ids)

test_data_prob = model.predict_proba(test_one_hot_tokens)
test_data_prob = pd.DataFrame(test_data_prob, columns=['EAP','HPL','MWS'])

print('model inference done')

test_result['EAP'] = test_data_prob['EAP']
test_result['HPL'] = test_data_prob['HPL']
test_result['MWS'] = test_data_prob['MWS']

test_result.to_csv('tmp/svm_ln_full.csv', index=False)
