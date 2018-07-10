#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:08:29 2018

@author: soojunghong

@about : Building POS tagger
@reference : https://nlpforhackers.io/training-pos-tagger/
"""

from nltk import word_tokenize, pos_tag

print pos_tag(word_tokenize("I am Soojung, I am learning NLP"))

#---------------------------------------------
# Picking a corpus to train the POS tagger
#---------------------------------------------
import nltk 

tagged_sentences = nltk.corpus.treebank.tagged_sents()

print tagged_sentences[0]
print "Tagged sentences: ", len(tagged_sentences)
print "Tagged words: ", len(nltk.corpus.treebank.tagged_words())

#--------------------------------------------------
# Training your own POS Tagger using scikit-learn
#--------------------------------------------------

#define feature 
# from following feature function take """ sentence: [w1, w2, ...], index: the index of the word """
def features(sentence, index):
    return {
            'word' : sentence[index],
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'is_capitalized': sentence[index][0].upper() == sentence[index][0],
            'is_all_caps': sentence[index].upper() == sentence[index],
            'is_all_lower': sentence[index].lower() == sentence[index],
            'prefix-1': sentence[index][0],
            'prefix-2': sentence[index][:2],
            'prefix-3': sentence[index][:3],
            'suffix-1': sentence[index][-1], # -1 means the index from the very right end
            'suffix-2': sentence[index][-2:], ## second right end 
            'suffix-3': sentence[index][-3:],
            'prev_word': '' if index == 0 else sentence[index-1],
            'next_word': '' if index == len(sentence)-1 else sentence[index+1],
            'has_hyphen': '-' in sentence[index],
            'is_numeric': sentence[index].isdigit(), 
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
            }

import pprint
pprint.pprint(features(['This', 'is', 'a', 'sentence'], 2))    


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

# build training set
# Split the dataset for training and testing 
cutoff = int(.75 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
training_sentences
test_sentences = tagged_sentences[cutoff:]
test_sentences 

print len(training_sentences)
print len(test_sentences)

def transform_to_dataset(tagged_sentences):
    X, y = [],[]
    
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])
    return X, y

X, y = transform_to_dataset(training_sentences)        
X
y

# train the classifier - Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer 
from sklearn.pipeline import Pipeline

clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', DecisionTreeClassifier(criterion='entropy'))
        ])
    
clf.fit(X[:10000], y[:10000]) #use first 10k samples

print 'Training completed'

X_test, y_test = transform_to_dataset(test_sentences)

print "Accuracy: ", clf.score(X_test, y_test)

# let's use classifier
def pos_tag(sentence):
    tagged_sentence = []
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])  
    return zip(sentence, tags)
        
print pos_tag(word_tokenize('this is my friend John'))