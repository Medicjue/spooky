#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:18:00 2017

@author: Julius
"""

from sklearn.svm import SVC
import pickle
import pandas as pd
import gc

random_seed = 23

f = open('data/histograms_k40.np', 'rb')
histograms = pickle.load(f)
f.close()

f = open('data/labels.np', 'rb')
labels = pickle.load(f)
f.close()

model = SVC(kernel="linear", C=1000, probability=True, random_state=random_seed)
model.fit(histograms, labels)

del(histograms)
del(labels)
gc.collect()

f = open('data/test_ids.np', 'rb')
test_ids = pickle.load(f)
f.close()

f = open('data/test_histograms_k40.np', 'rb')
test_histograms = pickle.load(f)
f.close()

test_result = pd.DataFrame()
test_result['id'] = pd.Series(test_ids)

test_data_prob = model.predict_proba(test_histograms)
test_data_prob = pd.DataFrame(test_data_prob, columns=['EAP','HPL','MWS'])

test_result['EAP'] = test_data_prob['EAP']
test_result['HPL'] = test_data_prob['HPL']
test_result['MWS'] = test_data_prob['MWS']

test_result.to_csv('tmp/d2v_svm_1118_k40.csv', index=False)
