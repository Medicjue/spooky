#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 22:35:06 2017

Exploratory Data
"""
import pandas as pd
import jieba

train_data = pd.read_csv('data/train.csv')

vocabs = dict()

for index, row in train_data.iterrows():
    text = row[1].lower()
    label = row[2]
    tokens = list(jieba.lcut(text))
#    print('{} - {}'.format(label, tokens))
    for token in tokens:
        vocabs[token] = vocabs.get(token, 0) + 1