#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:47:49 2020

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

#remove all the Punctuations and create list of list of words
def removepunctuation(reviews):
    all_reviews=list()
    for text in reviews:      
      text = "".join([ch for ch in text if ch not in punctuation])
      text = text.lower()
      all_reviews.append(text.split())
    return all_reviews


X,Y,_ = loadData()
xClean = removepunctuation(X)



all_text = " ".join(X)
all_words = all_text.split()
from collections import Counter 
# Count all the words using Counter Method
count_words = Counter(all_words)
total_words=len(all_words)
sorted_words=count_words.most_common(total_words)

vocab_to_int={w:i+1 for i,(w,c) in enumerate(sorted_words)}
encoded_reviews=list()
for review in X:
  encoded_review=list()
  for word in review.split():
    if word not in vocab_to_int.keys():
      #if word is not available in vocab_to_int put 0 in that place
      encoded_review.append(0)
    else:
      encoded_review.append(vocab_to_int[word])
  encoded_reviews.append(encoded_review)



sequence_length=250
features=np.zeros((len(encoded_reviews), sequence_length), dtype=int)
for i, review in enumerate(encoded_reviews):
  review_len=len(review)
  if (review_len<=sequence_length):
    zeros=list(np.zeros(sequence_length-review_len))
    new=zeros+review
  else:
    new=review[:sequence_length]
features[i,:]=np.array(new)



train_x=features[:int(0.8*len(features))]
train_y=Y[:int(0.8*len(features))]
valid_x=features[int(0.8*len(features)):int(0.9*len(features))]
valid_y=Y[int(0.8*len(features)):int(0.9*len(features))]
test_x=features[int(0.9*len(features)):]
test_y=Y[int(0.9*len(features)):]
print(len(train_y), len(valid_y), len(test_y))



max_seq_length = 250
vocab_size = 50 #50 is enough to vaoid the collitions
emb_out_len = 32
EPOC=50


model = Sequential()
model.add(Embedding(vocab_size, emb_out_len, input_length=max_seq_length))   #vocabsize, emb_out, inp len
#model.add(LSTM(emb_out_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(train_x, train_y, epochs=EPOC, verbose=0)

loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
print('Accuracy: %f' % (accuracy*100))









