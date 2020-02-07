#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:30:03 2020

@author: anandamohonghosh
"""


from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from string import punctuation
from collections import Counter 
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as utils
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Flatten
#from keras.layers.embeddings import Embedding
import pandas as pd
from string import punctuation
import numpy as np
import nltk
import math
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from torch import optim



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.embededLayer = nn.Embedding(num_embeddings =MAX_LENGTH, embedding_dim = VOCAB_SIZE, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
        self.lstmCells = nn.LSTM(VOCAB_SIZE, HIDDEN_IN, MAX_LENGTH)   #nn.LSTM(input_size, hidden_size, num_layers) 
        self.linearLayer = nn.Linear(128, 32)  # equivalent to Dense in keras
        self.dropOut = nn.Dropout(0.2)
        self.linearLayer2 = nn.Linear(32, 1) 
        self.reluAct = nn.ReLU()
        self.softAct = nn.Softmax()
        self.logSoftAct = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        clsf = self.embededLayer(x)
        clsf, _ = self.lstmCells(clsf)
        clsf = self.linearLayer(clsf[:,-1,:])
        clsf = self.reluAct(clsf)
        clsf = self.linearLayer2(clsf)
        clsf = self.sigmoid(clsf)
        return clsf



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


#remove all the Punctuations and create list words
def removepunctuation(reviews):
    all_reviews=list()
    for text in reviews:
        #text = text.lower()
        text = "".join([ch for ch in text if ch not in punctuation])
        #all_reviews.append(removeNonEng(text).split())
        all_reviews.append(removeNonEng(text))
    return all_reviews


def oneHotEncode(docs):
    encoded_docs = [one_hot(d, VOCAB_SIZE) for d in docs]
    return encoded_docs


def makePadded(encoded_docs):
    return pad_sequences(encoded_docs, maxlen=MAX_SENT_LENGTH, padding='post')
    #return padded_docs

    

#Static variables
MAX_LENGTH =100
MAX_SENT_LENGTH = 250
BATCH_SIZE = 32
VOCAB_SIZE = 50
HIDDEN_IN = 128
NUM_EPOCS = 1
LEARNINH_RATE = 0.003
WEIGHT_DECAY =1e-5


bOfWords = set(nltk.corpus.words.words())
X,Y,XT = loadData()
xClean = removepunctuation(X)
xTClean = removepunctuation(XT)
xOneHotEnc = oneHotEncode(xClean)
xPadded = makePadded(xOneHotEnc)
feature = torch.LongTensor(xPadded)

P = []
for x in Y:
    P.append(float(x))
Y = torch.FloatTensor(P)

trainDataset = utils.TensorDataset(feature, Y)
dataloaderTrain = utils.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=False)



availabelCuda =  torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Availabel CUDA ', availabelCuda)
clsf = Network()
model = clsf.to(availabelCuda)
optimizer = optim.Adam(model.parameters(), lr=LEARNINH_RATE,  weight_decay=WEIGHT_DECAY)
lossFunc= nn.BCELoss()



#Train model
print('Model Running on CUDA  ', availabelCuda)
for epoch in range(NUM_EPOCS):
    for xX, yY in dataloaderTrain:
        #print(xX.shape, yY.shape)
        dataX = Variable(xX).to(availabelCuda)
        dataY = Variable(yY).to(availabelCuda)
        outclass = clsf(dataX)
        optimizer.zero_grad()            # clear gradients for this training step
        loss = lossFunc(outclass, dataY)
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()
    print('Epoch: ', epoch+1, '| train loss: %.4f' % loss.data.numpy()) #CPU
        #print('Epoch: ', epoch+1, '| train loss: %.4f' % loss.item())  #GPU





'''
from nltk.tokenize import word_tokenize
#from nltk import sent_tokenize
import string

sentences = sent_tokenize(X)
tokens = word_tokenize(X[100])
words = [word for word in tokens if word.isalpha()]
tokens = [w.lower() for w in words]


table = X[100].maketrans('', '', string.punctuation)

stripped = [w.translate(table) for w in tokens]
words = [word for word in stripped if word.isalpha()]
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)

'''


















