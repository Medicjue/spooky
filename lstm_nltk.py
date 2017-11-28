#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:39:09 2017

LSTM + NLTK preprocessing + POS Tag
"""

import pandas as pd
import numpy as np
import gc
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import datetime

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import to_categorical

#from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

random_seed = 13

stop = set(stopwords.words('english'))

stemmer = SnowballStemmer('english')

punctuation = set([',', '.', ' ', '"', '\'', '?', '-', '+', '=', '!'])

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
    
def create_model(random_seed, vocab_size, max_sent_length):
    model = Sequential()
    model.add(Embedding(output_dim=32, input_dim=vocab_size, input_length=max_sent_length))
    model.add(Dropout(rate=0.2, seed=random_seed, name='Dropout'))
    model.add(LSTM(32))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(rate=0.2, seed=random_seed, name='2nd_dropout'))
    model.add(Dense(units=3, activation='softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

train_data = pd.read_csv('data/train.csv')

vocabs = set()
raw_tokens = []
sentences = []
labels = []
max_sent_length = 0

for index, row in train_data.iterrows():
    text = row[1].lower()
    sent_length = len(text.split(' '))
    if sent_length > max_sent_length:
        max_sent_length = sent_length
    sentences.append(text)
    label = row[2]
    tokens = work_prep(text)
    raw_tokens.append(list(tokens))
    vocabs |= tokens
    labels.append(label_map.index(label))



print('collect tokens done')


vocab_size = len(vocabs)

tokenizer = Tokenizer(vocab_size)

tokenizer.fit_on_texts(sentences)

x_train_seq = tokenizer.texts_to_sequences(sentences)

x_train = sequence.pad_sequences(x_train_seq, max_sent_length)

model = create_model(random_seed, vocab_size, max_sent_length)

y_train = to_categorical(labels)

histroys = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.1)

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

start = datetime.datetime.now()
#model = RandomForestClassifier(n_jobs=4, random_state=random_seed)
model = SVC(kernel="linear", C=1000, probability=True, random_state=random_seed)
model.fit(one_hot_tokens, labels)
end = datetime.datetime.now()
print('model training done, time consume:{}'.format(end-start))
test_data = pd.read_csv('data/test.csv')

test_sentences = []
ids = []
test_one_hot_tokens = []
for index, row in test_data.iterrows():
    ids.append(row[0])
    text = row[1].lower()
    test_sentences.append(text)
    
    """
    ids.append(row[0])
    text = row[1].lower()
#    tokens = set(jieba.lcut(text))
    tokens = work_prep(text)
    one_hot_token = np.zeros(vocab_size+1, dtype=bool)
    for token in tokens:
        try:
            one_hot_token[vocabs.index(token)] = True
        except:
            ## Mark as others
            one_hot_token[vocab_size] = True
    test_one_hot_tokens.append(one_hot_token)
    """
    
x_test_seq = tokenizer.texts_to_sequences(test_sentences)

x_test = sequence.pad_sequences(x_test_seq, max_sent_length)

predict_y = model.predict_classes(x_test)
predict_y_prob = model.predict(x_test)
    
gc.collect()
print('convert test data 2 one-hot done')

test_result = pd.DataFrame()
test_result['id'] = pd.Series(ids)

test_data_prob = model.predict_proba(test_one_hot_tokens)
test_data_prob = pd.DataFrame(predict_y_prob, columns=['EAP','HPL','MWS'])

print('model inference done')

test_result['EAP'] = test_data_prob['EAP']
test_result['HPL'] = test_data_prob['HPL']
test_result['MWS'] = test_data_prob['MWS']

test_result.to_csv('tmp/lstm_nltk_1epoch.csv', index=False)
