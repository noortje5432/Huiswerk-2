# -*- coding: utf-8 -*-
"""
File: evaluate_models.py
Created on Thu Jun  8 12:13:13 2023

@author: Jasmijn Lie

Loads and evaluates models with test data and saves the output in a evaluation_output.txt file
"""

import pickle
from nltk.corpus import conll2002 as conll

testdata = conll.chunked_sents("ned.testa") #Test data

#best.pickle
best = pickle.load(open("best.pickle", "rb"))
print("New_features performance:\n", best.accuracy(testdata))

#other.pickle
other = pickle.load(open("other.pickle", "rb"))
print("Other_new_features1 performance:\n", other.accuracy(testdata))

#another.pickle
another = pickle.load(open("another.pickle", "rb"))
print("Other_new_features2 performance:\n", another.accuracy(testdata))

#Saved output put in an text file
with open("evaluation_output.txt", 'w') as f:
    f.write("The best.pickle: New_features\n" + str(best.accuracy(testdata)))
    f.write("\n")
    f.write("\nOther_new_features1 performance:\n" + str(other.accuracy(testdata)))
    f.write("\n")
    f.write("\nOther_new_features2 performance:\n" + str(another.accuracy(testdata)))
