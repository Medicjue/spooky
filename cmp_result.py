#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 00:00:23 2017

Compare result
"""

import pandas as pd

result_map = ['EAP','HPL','MWS']

fastText_result = pd.read_csv('tmp/fastText_result.csv')
svm_result = pd.read_csv('tmp/test_result_1115.csv')

fastText_ans = []
for index, row in fastText_result.iterrows():
    base = float(row[1])
    i = 0
    if float(row[2]) > base:
        base = float(row[2])
        i = 1
    if float(row[3]) > base:
        base = float(row[3])
        i = 2
    fastText_ans.append(result_map[i])
    

svm_ans = []
for index, row in svm_result.iterrows():
    base = float(row[1])
    i = 0
    if float(row[2]) > base:
        base = float(row[2])
        i = 1
    if float(row[3]) > base:
        base = float(row[3])
        i = 2
    svm_ans.append(result_map[i])
    
cmp = pd.DataFrame()
cmp['ft'] = pd.Series(fastText_ans)
cmp['svm'] = pd.Series(svm_ans)

diff_cmp = cmp[cmp['ft'] != cmp['svm']]
