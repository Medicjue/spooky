#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 07:11:59 2017

classifier comparison
"""

import pandas as pd
import numpy as np
import gc
import pickle
import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

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
train_size = int(ttl_size * 0.34)

train_x = one_hot_tokens[0:train_size]
train_y = labels[0:train_size]

test_x = one_hot_tokens[train_size:]
test_y = labels[train_size:]

result_cmp = []

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=1000),
    SVC(gamma=2, C=1000),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
for name, clf in zip(names, classifiers):
    train_start = datetime.datetime.now()
    clf.fit(train_x, train_y)
    train_end = datetime.datetime.now()
    train_ttl = train_end - train_start
    
    predict_start = datetime.datetime.now()
    predict_y = clf.predict(test_x)
    predict_end = datetime.datetime.now()
    predict_ttl = predict_end - predict_start
    
    t1, t2, t3, t4, t5, t6, t7, t8, t9 = confusion_matrix(test_y, predict_y).ravel()
    
    result = pd.DataFrame()
    result['actual'] = pd.Series(test_y)
    result['predict'] = pd.Series(predict_y)
    
    diff_gap = result[result['actual'] != result['predict']]
    
    result_cmp.append([name, len(diff_gap), t1, t2, t3, t4, t5, t6, t7, t8, t9,  str(train_ttl), str(predict_ttl)])
    
    f = open('mdl/{}.model'.format(name), 'wb')
    pickle.dump(clf, f)
    f.close()
    
    del(result)
    gc.collect()
    
result_cmp = pd.DataFrame(result_cmp, columns=['Model', 'Diff', 'True-EAP', 'False-MWS-butEAP', 'False-HPL-butEAP', 'False-EAP-butMWS', 'True-MWS', 'False-HPL-butMWS', 'False-EAP-butHPL', 'False-MWS-butHPL', 'True-HPL', 'trainTime', 'predictTime'])
result_cmp.to_csv('tmp/classifier_cmp.csv', index=False)