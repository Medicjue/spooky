#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 23:45:39 2017

Convert fastText result
"""

import pandas as pd


EAP = []
HPL = []
MWS = []

f = open('tmp/test_predict.txt', 'r')
for line in f:
    result = line[9:12]
    prob =  float(line[13:])
    rest_prob = (1 - prob)/2
    if result == 'EAP':
        EAP.append(prob)
        HPL.append(rest_prob)
        MWS.append(rest_prob)
    elif result == 'HPL':
        EAP.append(rest_prob)
        HPL.append(prob)
        MWS.append(rest_prob)
    elif result == 'MWS':
        EAP.append(rest_prob)
        HPL.append(rest_prob)
        MWS.append(prob)
f.close()

len(EAP)
len(HPL)
len(MWS)

test_data = pd.read_csv('data/test.csv')

test_ids = []

for index, row in test_data.iterrows():
    test_id = row[0]
    test_ids.append(test_id)
len(test_ids)

result = pd.DataFrame()
result['id'] = pd.Series(test_ids)
result['EAP'] = pd.Series(EAP)
result['HPL'] = pd.Series(HPL)
result['MWS'] = pd.Series(MWS)
result.to_csv('tmp/fastText_result.csv', index=False)