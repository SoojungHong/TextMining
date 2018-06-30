#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 15:02:10 2018

@author: soojunghong

@about : Implementation example of Non-Negative Matrix Factorization 
"""

---------------------------------------------------------------------------------------------------------------------------
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
allData

data_text = allData[['headline_text']]
data_text = data_text.astype('str')
data_text.iloc[0:].values #iloc[0:] means all row start from 0th row  (from row index : end row index)
train_headlines = [value[0] for value in data_text.iloc[0:].values];
train_headlines

# Count Vectorizer module needs string inputs, not array, so join them with a space
train_headlines_sentences = [''.join(text) for text in train_headlines]
train_headlines_sentences

vectorizer = CountVectorizer(analyzer='word', max_features=5000)
x_counts = vectorizer.fit_transform(train_headlines_sentences)
x_counts # compressed Sparse Row format 

transformer = TfidfTransformer(smooth_idf = False)
x_tfidf = transformer.fit_transform(x_counts)
x_tfidf # compressed sparse row format
x_tfidf_norm = normalize(x_tfidf, norm ='l1', axis=1) #normalize the TF-IDF values to unit length for each row 

# obtain NMF model and fit it with sentences 
num_topics = 10
model = NMF(n_components = num_topics, init='nndsvd')
model.fit(x_tfidf_norm) # fit model

def get_nmf_topics(model, n_top_words):
    feat_names = vectorizer.get_feature_names()
    word_dict = {}
    for i in range(num_topics):
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words
    
    return pd.DataFrame(word_dict)              

get_nmf_topics(model, 20)