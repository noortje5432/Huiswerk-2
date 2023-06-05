# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 18:12:24 2023

@author: jasmi
"""

def test_features(sentence, i, history):
    """dummy Chunker features designed to test the Chunker class for correctness
        - the POS tag of the word
        - the entire history of IOB tags so far
            formatted as a tuple because it needs to be hashable
    """ 
    word, pos = sentence[i]
    return { 
        "pos": pos,
        "whole history": tuple(history)
            }