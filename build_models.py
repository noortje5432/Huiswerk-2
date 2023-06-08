# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 18:38:11 2023

@author: jasmi

"""
import nltk
from custom_chunker import ConsecutiveNPChunker
from features import new_features 
from pickle import dump
from nltk.corpus import conll2002 as conll

traindata = conll.chunked_sents("ned.train")

new_test = ConsecutiveNPChunker(new_features, traindata )
with open("test.pickle", "wb") as output: #other.pickle even test.pickle genoemd 
    dump(new_test, output)

Naive_Bayes = nltk.NaiveBayesClassifier.train(traindata)
with open("best.pickle", "wb") as output:
    dump(Naive_Bayes, output)
     
