#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:08:19 2018

@author: soojunghong
"""

import pandas as pd

# Basic Feature Extraction 
path_data = "/Users/soojunghong/Documents/safariML/ML_python/"
csv_file = "train.csv"
train = pd.read_csv(path_data + csv_file)
type(train)
train['word_count'] = train['SentimentText'].apply(lambda x: len(str(x).split(" "))) #apply(function) - apply function along an axis 
train['word_count'] # dataframe train has one more column called 'word_count'
train[['SentimentText', 'word_count']].head()

# Number of characters
train['char_count'] = train['SentimentText'].str.len() #include spaces
train['char_count']
train[['SentimentText','char_count']].head()

# average word length 
def avg_word(sentence):
    words =  sentence.split()
    return (sum(len(word) for word in words)/len(words))

train['avg_word'] = train['SentimentText'].apply(lambda x : avg_word(x))
train[['SentimentText', 'avg_word']].head()

# Number of stopwords 
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['stopwords'] = train['SentimentText'].apply(lambda x : len([x for x in x.split() if x in stop]))
train[['SentimentText', 'stopwords']].head()

# Number of special character 
train['hashtags'] = train['SentimentText'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['SentimentText', 'hashtags']].head()

# Is number 
train['numerics'] = train['SentimentText'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['SentimentText', 'numerics']].head()

# removing punctuation - punctuation is the marks, such as full stop, comma, and brackets, used in writing to separate sentences and their elements and to clarify meaning.
train['SentimentText'] = train['SentimentText'].str.replace('[^\w\s]', '')
train['SentimentText'].head()


# stopword : a stop word is a commonly used word (such as "the") that a search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query.
# removing stop words 
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['SentimentText'] = train['SentimentText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['SentimentText'].head()