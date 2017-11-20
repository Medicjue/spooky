#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 23:49:07 2017

fasttext
"""

import pandas as pd
import jieba


train_data = pd.read_csv('data/train.csv')


f = open('tmp/train.txt', 'w')

for index, row in train_data.iterrows():
    text = row[1].lower()
    label = row[2]
    tokens = set(jieba.lcut(text))
    f.write('__label__{} {}'.format(label, ' '.join(tokens)))
    
f.close()
    
f = open('tmp/test.txt', 'w')
test_data = pd.read_csv('data/test.csv')
for index, row in test_data.iterrows():
    text = row[1].lower()
    tokens = set(jieba.lcut(text))
    f.write('{}'.format(' '.join(tokens)))
    
f.close()