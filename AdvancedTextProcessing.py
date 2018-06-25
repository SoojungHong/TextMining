#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:58:59 2018

@author: soojunghong

@About : advanced Text processing to extract features using NLP Techniques 
"""

# N-grams : N = 1 (unigram), N = 2 (bigram), N = 3 (trigram) 
# Unigram usually don't contain much information compared to bigram, trigram
# n-gram is try to capture the language structure, like what letter or word is likely to follow the given one
# Because the longer the n-gram (higher n), the more context you have to work with
# optimum length depends on the application

import pandas as pd

path_data = "/Users/soojunghong/Documents/safariML/ML_python/TextMining/"
csv_file = "train.csv"
train = pd.read_csv(path_data + csv_file)

train['SentimentText'][0]
TextBlob(train['SentimentText'][0]).ngrams(2)

# Term frequency 
# The ratio of the count of a word present in a sentence, to the length of the sentence
# TF = (Number of times term T appear in the particular row)/(number of terms in that row)
train['SentimentText'][1:2]
(train['SentimentText'][1:2]).apply(lambda x: pd.value_counts(x.split(" ")))
(train['SentimentText'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)


tf1 = (train['SentimentText'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words', 'tf']
tf1.columns
tf1

# Inverse Document Frequency 
# word is not of much use to us if it is appearing in all the documents
# the intuition behind the Inverse Document Frequency (IDF) is that a word is not much of use if it appearing in all the couments

# it is the ratio of the total number of rows to the number of rows in which that word is present
# IDF = log(N/n), where N is the total number of rows and n is the number of rows in which the word was present. 

import numpy as np
for i, word in enumerate(tf1['words']): #enumerate
    tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['SentimentText'].str.contains(word)])))
    

tf1