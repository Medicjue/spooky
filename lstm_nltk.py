#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:39:09 2017

LSTM + NLTK preprocessing + POS Tag
"""

import pandas as pd
import gc
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import datetime

from sklearn.utils import shuffle

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import to_categorical

random_seed = 13

train_epochs = 10

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

start = datetime.datetime.now()
train_data = pd.read_csv('data/train.csv')

vocabs = set()
raw_tokens = []
all_sentences = []
train_sentences = []
labels = []
max_sent_length = 0

for index, row in train_data.iterrows():
    text = row[1].lower()
    sent_length = len(text.split(' '))
    if sent_length > max_sent_length:
        max_sent_length = sent_length
    all_sentences.append(text)
    train_sentences.append(text)
    label = row[2]
    tokens = work_prep(text)
    raw_tokens.append(list(tokens))
    vocabs |= tokens
    labels.append(label_map.index(label))

test_data = pd.read_csv('data/test.csv')

test_sentences = []
ids = []
for index, row in test_data.iterrows():
    ids.append(row[0])
    text = row[1].lower()
    sent_length = len(text.split(' '))
    if sent_length > max_sent_length:
        max_sent_length = sent_length
    all_sentences.append(text)
    test_sentences.append(text)
    tokens = work_prep(text)
    raw_tokens.append(list(tokens))
    vocabs |= tokens

end = datetime.datetime.now()
collect_time = end - start

print('collect sentences done, time consume:{}'.format(collect_time))
gc.collect()

start = datetime.datetime.now()
vocab_size = len(vocabs)

tokenizer = Tokenizer(vocab_size)

tokenizer.fit_on_texts(all_sentences)

x_train_seq = tokenizer.texts_to_sequences(train_sentences)

x_train = sequence.pad_sequences(x_train_seq, max_sent_length)

y_train = to_categorical(labels)

x_test_seq = tokenizer.texts_to_sequences(test_sentences)

x_test = sequence.pad_sequences(x_test_seq, max_sent_length)

end = datetime.datetime.now()
prepare_time = end - start

print('prepare train / test data done, time consume:{}'.format(prepare_time))
gc.collect()

model = create_model(random_seed, vocab_size, max_sent_length)

for epoch in range(train_epochs):
    x_train, y_train = shuffle(x_train, y_train, random_state=epoch)
    gc.collect()
    historys = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.1)
    print(historys)
    model.save('tmp/lstm_epoch_{}.model'.format(epoch))
    gc.collect()


predict_y = model.predict_classes(x_test)
predict_y_prob = model.predict(x_test)


test_result = pd.DataFrame()
test_result['id'] = pd.Series(ids)

test_data_prob = pd.DataFrame(predict_y_prob, columns=['EAP','HPL','MWS'])

print('model inference done')

test_result['EAP'] = test_data_prob['EAP']
test_result['HPL'] = test_data_prob['HPL']
test_result['MWS'] = test_data_prob['MWS']

test_result.to_csv('tmp/lstm_nltk_{}_epoch.csv'.format(train_epochs), index=False)
