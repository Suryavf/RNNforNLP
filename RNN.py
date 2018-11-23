#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:23:21 2018

@author: victor
"""


"""
Import librarys
---------------
"""
import os.path
import numpy  as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pyprind

"""
Import Data
-----------
"""
print('Import Data')
stop = stopwords.words('english')
porter = PorterStemmer()

def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label   
            
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text

def get_minibatch(doc_stream, size):
    docs, y = [], []
    for _ in range(size):
        text, label = next(doc_stream)
        docs.append(text)
        y.append(label)
    return docs, y


"""
Preprocessing
-------------
"""
def preprocessing(text):
    import re
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\n)")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    text = REPLACE_NO_SPACE.sub('', text.lower())
    text = REPLACE_WITH_SPACE.sub(' ', text)
    return text


"""
Parameters
----------
"""
n_samples = 50000
n_train   = 40000
n_test    = n_samples - n_train
n_vector  = 100
n_tokens  = 180

fdata  = 'shuffled_movie_data.csv'
fmodel = "word2vec.model"

"""
Load word2vect
--------------

from gensim.models import Word2Vec
doc_stream = stream_docs(path=fdata)

common_texts, sentiment = get_minibatch(doc_stream, size=n_samples)
common_texts = [ preprocessing(common_texts[n]).split() for n in range(n_samples) ]
    
# Save/Read model
if not os.path.isfile(fmodel):
    print('Creando modelo')
    model = Word2Vec(common_texts, size=n_vector, 
                     window=10, min_count=1, workers=4)
    model.save(fmodel)

else:
    print('Cargando modelo')
    model = Word2Vec.load(fmodel)
"""

"""
Parameters
----------
"""
n_epoch  = 500
n_batch  = 100
n_hidden = 10
n_vocab  = len(model.wv.vocab)
dropout  = 0.2


"""
Feature extraction
------------------
"""
def get_sequence(b):
    start = b* n_batch
    
    # Batch
    X = np.zeros( [n_batch,n_tokens,n_vector] )
    y = np.array(sentiment[start:start+n_batch]).reshape([1,n_batch])
    
    for n in range(n_batch):
        # New sample
        words = common_texts[start+n]; n_words = len(words); 
        if n_words > n_tokens: n_words = n_tokens;
        
        # Model
        for w in range(n_words):
            X[n][w] = model.wv[ words[w] ]
        
    return X,y


"""
Keras setup
-----------
"""
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

rnn = Sequential()

# Embeding
rnn.add( Bidirectional( LSTM( n_hidden, input_shape=(n_vector,1), 
                              return_sequences=True,
                              dropout=dropout,
                              recurrent_dropout=dropout,
                              kernel_initializer = 'he_normal',
                                bias_initializer = 'he_normal') ) )
rnn.add( Bidirectional( LSTM( n_hidden,
                              return_sequences=True,
                              dropout=dropout,
                              recurrent_dropout=dropout,
                              kernel_initializer = 'he_normal',
                                bias_initializer = 'he_normal') ) )
rnn.add( Bidirectional( LSTM( n_hidden, 
                              return_sequences=True,
                              dropout=dropout,
                              recurrent_dropout=dropout,
                              kernel_initializer = 'he_normal',
                                bias_initializer = 'he_normal') ) )
rnn.add( TimeDistributed(Dense( 1,         activation = 'sigmoid'  ,
                   kernel_initializer = 'he_normal',
                     bias_initializer = 'he_normal' ) ) )
rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


"""
Train recurrent neural network 
------------------------------
"""
print('\n\n')
print('Train RNN')
print('---------')
for epoch in range(n_epoch):
    for b in range(int(n_train/n_batch)):
    	# generate new random sequence
    	x,y = get_sequence(b)
        
    	# fit model for one epoch on this sequence
    	rnn.fit(x, y, epochs=1, batch_size=n_batch, verbose=2)
        