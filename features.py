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

def new_features(sentence, i, history):
    """Features:
        - the POS tag of the word
        - full_capital: Full capital word
        - pos_next_word: pos of the next word
        - pos_previous_word: pos of the previous word
        - dutch_name: checks if there is a dutch name
        - first_capital: checks if first letter is capital
        - stripe: checks for '-'
        - previous_word: previous word
        - previous_word_cap: checks if the previous word's first letter is a capital
        - word: the word itself
        - mv: word that ends with 's'       
    """ 
    import csv

    conn = open("VNC2013.csv", encoding="utf-8")
    myreader = csv.reader(conn)
    raw_data = [ row for row in myreader ]
    conn.close()
    
    names = []
    for name, MV, amount in raw_data:
        if int(amount) > 1:
            names.append(name)

    word, pos = sentence[i]
    if word.isupper():
        full_capital = True
    else: 
        full_capital = False
        
    if i+1 < len(sentence):
        pos_next_word = sentence[i+1][1] 
    else: 
        pos_next_word = None
        
    if i-1 > 0:
        pos_previous_word = sentence[i-1][1]
    else: 
        pos_previous_word = None
        
    if word in names:
        dutch_name = True
    else:
        dutch_name = False
        
    if word[0].isupper():
        first_capital = True
    else: 
        first_capital = False

    if "-" in word:
        stripe = True
    else:
        stripe =False
        
    if i-1 > 0:
        previous_word = sentence[i-1][0]
    else: 
        previous_word = None
        
    if i-1 > 0:
        if sentence[i-1][0][0].isupper():
            previous_word_cap = True
        else: 
            previous_word_cap = False
    else: 
        previous_word_cap = False
        
    if len(word) > 1:
        if word[-1] == "s":
            mv = True
        else:
            mv = False
    else: 
        mv = False
        
        
    return { 
        "pos": pos, 
        "all capitals": full_capital,
        "pos next word" : pos_next_word,
        "pos previous word" : pos_previous_word,
        "is dutch name" : dutch_name,  
        "first letter is capital" : first_capital,
        "stripe in word" : stripe,
        "previous word" : previous_word, 
        "previous word has capital" : previous_word_cap,
        "word" : word,
        "word ends with 's'" : mv,
            }