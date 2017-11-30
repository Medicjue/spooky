#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:30:13 2017

baseline SVM
"""

import pandas as pd
import numpy as np
import gc
import datetime

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.svm import SVC

random_seed = 13

stop = set(stopwords.words('english'))

stemmer = SnowballStemmer('english')

punctuation = set([',', '.', ' ', '"', '\'', '?', '-', '+', '=', '!', '~', '/', ':', ';', '@', '#', '$', '%', '&', '*', '\n'])

label_map = ['EAP', 'MWS', 'HPL']


def work_prep(sentence):
    words = word_tokenize(text)
    tokens = set(words)
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
    
vocabs = vocabs - punctuation
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

#model = RandomForestClassifier(n_jobs=4, random_state=random_seed)
print('Start train SVM model')
start = datetime.datetime.now()
model = SVC(kernel="linear", C=1000, probability=True, random_state=random_seed)
model.fit(one_hot_tokens, labels)
end = datetime.datetime.now()
print('End training, time consume: {}'.format(end-start))

test_data = pd.read_csv('data/test.csv')

ids = []
test_one_hot_tokens = []
for index, row in test_data.iterrows():
    ids.append(row[0])
    text = row[1].lower()
    tokens = work_prep(text)
    one_hot_token = np.zeros(vocab_size+1, dtype=bool)
    for token in tokens:
        try:
            one_hot_token[vocabs.index(token)] = True
        except:
            ## Mark as others
            one_hot_token[vocab_size] = True
    test_one_hot_tokens.append(one_hot_token)
    
gc.collect()

test_result = pd.DataFrame()
test_result['id'] = pd.Series(ids)

test_data_prob = model.predict_proba(test_one_hot_tokens)
test_data_prob = pd.DataFrame(test_data_prob, columns=['EAP','HPL','MWS'])

test_result['EAP'] = test_data_prob['EAP']
test_result['HPL'] = test_data_prob['HPL']
test_result['MWS'] = test_data_prob['MWS']

test_result.to_csv('tmp/svm_nltk.csv', index=False)
