# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:13:13 2023

@author: jasmi
"""

import pickle
from nltk.corpus import conll2002 as conll

testdata = conll.chunked_sents("ned.testa")

#best.pickle
other = pickle.load(open("best.pickle", "rb"))
print("New_features performance:", other.accuracy(testdata))

#other.pickle
best = pickle.load(open("other.pickle", "rb"))
print("Naive Bayes Classifier:", best.accuracy(testdata))
