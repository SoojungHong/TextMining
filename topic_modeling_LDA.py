#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 19:17:05 2018

@author: soojunghong
"""
#---------------------------------------------------------------------------------------------------------------------------
# Topic Modeling 
# https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df
#---------------------------------------------------------------------------------------------------------------------------

import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
import pickle;


allData = pd.read_csv('/Users/soojunghong/Documents/safariML/ML_python/kaggle/InstacartAnalysis/abcnews-date-text.csv', error_bad_lines=False)
type(allData)

# get only 1000 rows of dataframe, since it takes too much time to remove stop words
nRows = 1000
data = allData[0:nRows]
data

data_text = data[['headline_text']]; # we only need text column from the original data
data_text = data_text.astype('str')
data_text

for idx in range(len(data_text)):
    #remove stop words
    data_text.iloc[idx]['headline_text'] = [word for word in data_text.iloc[idx]['headline_text'].split(' ') if word not in stopwords.words()]
    
    #print logs to monitor output
    if idx % 1000 == 0:
        sys.stdout.write('\rc = ' + str(idx) + ' / ' + str(len(data_text)));

#save data because it takes a while to remove stop words 
pickle.dump(data_text, open('data_text.dat', 'wb'))      

data_text # array of string that not stop words 
#get words as an array for LDA input
data_text.iloc[0:]
type(data_text.iloc[0:].values) # narray type 

for value in data_text.iloc[0:].values:
    print(value)

train_headlines = [value[0] for value in data_text.iloc[0:].values];  
train_headlines

num_topics = 10
id2word = gensim.corpora.Dictionary(train_headlines)
corpus = [id2word.doc2bow(text) for text in train_headlines] #doc2bow : The function doc2bow() simply counts the number of occurrences of each distinct word, converts the word to its integer word id and returns the result as a sparse vector.

lda = ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

def get_lda_topics(model, num_topics):
    word_dict = []
    for i in range(num_topics):
        words = model.show_topic(i, topn=20)
        
        for i in words : 
            print i[0]
            word_dict.append(i[0])
        #word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words]; #format(i+1) is to start topic index from 1 (since 0 is CS people' index )
    return pd.DataFrame(word_dict)

get_lda_topics(lda, num_topics)   

