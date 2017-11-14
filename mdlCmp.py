#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 00:15:16 2017

Classifier Model Comparison
"""

import pandas as pd
import jieba
import numpy as np
import gc
import pickle

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



random_seed = 23

def prepare_models(random_seed):
    models = dict()
    rf_model = RandomForestClassifier(n_jobs=4, random_state=random_seed)
    dt_model = DecisionTreeClassifier(max_depth=10)
    ada_model = AdaBoostClassifier()
    models['RandomForest'] = rf_model
    models['DecisionTree'] = dt_model
    models['AdaBoost'] = ada_model
    return models

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

comparison = []

models = prepare_models(random_seed)
for key, value in models.items():
    model_name = key
    model = value
    
    model.fit(train_x, train_y)

    predict_y = model.predict(test_x)
    predict_y_prob = model.predict_proba(test_x)
    
    result = pd.DataFrame()
    result['actual'] = pd.Series(test_y)
    result['predict'] = pd.Series(predict_y)
    
    match_cnt = len(result[result['actual'] == result['predict']])
    mismatch_cnt = len(result[result['actual'] != result['predict']])
    
    f = open('mdl/'+model_name+'.model', 'wb')
    pickle.dump(model, f)
    f.close()

    comparison.append([model_name, match_cnt, mismatch_cnt])

comparison = pd.DataFrame(comparison, columns=['Model','Match','Mismatch'])
comparison.to_csv('tmp/mdl_cmp.csv', index=False)