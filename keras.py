#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:51:13 2020

@author: anandamohonghosh
"""

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


df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
text = np.asarray(df)
sentiment = text[:, 4]
reviews = text[:, 3]

#Special symbols are removed using the punctuations
##$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
all_reviews=list()
for text in reviews:
  #text = text.lower()
  text = "".join([ch for ch in text if ch not in punctuation])
  all_reviews.append(text)
all_text = " ".join(all_reviews)
all_words = all_text.split()

vocab_size = 250
encoded_docs = [one_hot(d, vocab_size) for d in all_reviews]
print(encoded_docs)






# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.',
        'Our Deeds are the Reason of this earthquake May ALLAH Forgive us all']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0,0])
# integer encode the documents
vocab_size = 250
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)



# pad documents to a max length of 4 words
max_length = 64
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))