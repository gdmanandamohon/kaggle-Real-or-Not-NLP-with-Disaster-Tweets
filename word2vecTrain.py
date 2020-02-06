#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:35:32 2020

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
import nltk


def loadData():
    df = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    text = np.asarray(df)
    sentiment = text[:, 4]
    reviews = text[:, 3]
    
    text_t = np.asarray(df_test)
    reviews_t = text_t[:, 3]
    
    return reviews, sentiment, reviews_t



def removeNonEng(s):
    return " ".join(w for w in nltk.wordpunct_tokenize(s) if w.lower() in bOfWords or not w.isalpha())


#remove all the Punctuations and create list of list of words
def removepunctuation(reviews):
    all_reviews=list()
    for text in reviews:      
      text = "".join([ch for ch in text if ch not in punctuation])
      text = text.lower()
      all_reviews.append(removeNonEng(text).split())
    return all_reviews


def oneHotEncode(docs):
    vocab_size = 50
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    return encoded_docs


def makePadded(encoded_docs):
    max_length = 64
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return padded_docs

    


bOfWords = set(nltk.corpus.words.words())
X,Y,XT = loadData()
xClean = removepunctuation(X)
xTClean = removepunctuation(XT)


model = Word2Vec(xClean, min_count=1, size= 50, workers=5, window =5, sg = 1) #sg is skip-gram, if not then CBOW    


encoded_docs = [[model.wv[word] for word in post] for post in xClean]

import math
MAX_LENGTH =100

padded_posts = []

for post in encoded_docs:
    # Pad short posts with alternating min/max
    if len(post) < MAX_LENGTH:
        
        # Method 1
        #pointwise_min = np.minimum.reduce(post)
        #pointwise_max = np.maximum.reduce(post)
        #padding = [pointwise_max, pointwise_min]
        
        # Method 2
        pointwise_avg = np.mean(post)
        padding = [pointwise_avg]
        
        post += padding * math.ceil((MAX_LENGTH - len(post) / 2.0))
        
    # Shorten long posts or those odd number length posts we padded to 51
    if len(post) > MAX_LENGTH:
        post = post[:MAX_LENGTH]
    
    # Add the post to our new list of padded posts
    padded_posts.append(post)












