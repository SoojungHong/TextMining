#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 18:06:51 2018

@author: soojunghong

@k-means algorithm for English 

@About : k-means creates k groups from a set of objects so that the members of a group are more similar. It’s a popular cluster analysis technique for exploring a dataset.
"""

from numpy import array
from numpy import random
from scipy.cluster.vq import vq, kmeans, whiten 

features =  array([[ 1.9,2.3],
                    [ 1.5,2.5],
                    [ 0.8,0.6],
                    [ 0.4,1.8],
                    [ 0.1,0.1],
                    [ 0.2,1.8],
                    [ 2.0,0.5],
                    [ 0.3,1.5],
                    [ 1.0,1.0]])

features

# whiten : Normalize a group of observations on a per feature basis.
#          Before running k-means, it is beneficial to rescale each feature dimension of the observation set with whitening. Each feature is divided by its standard deviation across all observations to give it unit variance.

whitened = whiten(features)
whitened[0]
book = array((whitened[0],whitened[2]))
book
kmeans(whitened,book)

random.seed((1000, 2000)) 
code = 3 
kmeans(whitened, code)