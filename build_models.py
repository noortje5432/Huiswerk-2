# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 18:38:11 2023

@author: jasmi

"""
import nltk
from custom_chunker import ConsecutiveNPChunker
from features import new_features, other_new_features1, other_new_features2  
import pickle 
from nltk.corpus import conll2002 as conll

traindata = conll.chunked_sents("ned.train") # Train_data

new_test = ConsecutiveNPChunker(new_features, traindata ) # new_features model, beste model
with open("best.pickle", "wb") as output: 
    pickle.dump(new_test, output)

other_test = ConsecutiveNPChunker(other_new_features1, traindata) # other_new_features1 model
with open("other.pickle", "wb") as output:
    pickle.dump(other_test, output)
    
other_test2 = ConsecutiveNPChunker(other_new_features2, traindata) # other_new_features2 model
with open("another.pickle", "wb") as output:
    pickle.dump(other_test2, output)
     
