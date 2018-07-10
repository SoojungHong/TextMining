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
            'suffix-1': sentence[index][-1], # ??
            'suffix-2': sentence[index][-2:], ## ??
            'suffix-3': sentence[index][-3:],
            'prev_word': '' if index == 0 else sentence[index-1],
            'next_word': '' if index == len(sentence)-1 else sentence[index+1],
            'has_hyphen': '-' in sentence[index],
            'is_numeric': sentence[index].isdigit(), 
            'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
            }

import pprint
pprint.pprint(features(['This', 'is', 'a', 'sentence'], 2))    
