#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:30:12 2020

@author: anandamohonghosh
"""


from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from string import punctuation
from collections import Counter 
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from keras.layers import LSTM
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import pandas as pd
from string import punctuation
import numpy as np
import nltk

def loadData():
    df = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    text = np.asarray(df)
    sentiment = text[:, 4]
    reviews = text[:, 3]
    
    text_t = np.asarray(df_test)
    #sentiment_t = text_t[:, 4]
    reviews_t = text_t[:, 3]
    
    return reviews, sentiment, reviews_t



def oneHotEncode(docs):
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    #print(encoded_docs)
    return encoded_docs


def makePadded(encoded_docs): 
    padded_docs = pad_sequences(encoded_docs, maxlen=max_seq_length, padding='post')
    #print(padded_docs)
    return padded_docs


def removeNonEng(s):
    return " ".join(w for w in nltk.wordpunct_tokenize(s) if w.lower() in bOfWords or not w.isalpha())


def removepunctuation(reviews):
    all_reviews=list()
    for text in reviews:      
      text = "".join([ch for ch in text if ch not in punctuation])
      text = text.lower()
      all_reviews.append(removeNonEng(text).split())
    return all_reviews
    

#Inputs
    
bOfWords = set(nltk.corpus.words.words())
max_seq_length = 25
vocab_size = 50 #50 is enough to vaoid the collitions
emb_out_len = 32
EPOC=50

X,Y,X_T = loadData()
X = removepunctuation(X)
xOneHot = oneHotEncode(X)
paddedMat = makePadded(xOneHot)

xOneHot = oneHotEncode(X)
paddedMat = makePadded(xOneHot)

xTOneHot = oneHotEncode(X_T)
paddedMat_T = makePadded(xTOneHot)


'''

model = Sequential()
model.add(Embedding(vocab_size, emb_out_len, input_length=max_seq_length))   #vocabsize, emb_out, inp len
model.add(LSTM(emb_out_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(paddedMat, Y, epochs=EPOC, verbose=0)
loss, accuracy = model.evaluate(paddedMat, Y, verbose=0)
print('Accuracy: %f' % (accuracy*100))
'''




















