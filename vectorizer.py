#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 03:06:33 2017

@author: chandu
"""

from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem.porter import PorterStemmer
import re
import pickle
import os

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    porter = PorterStemmer()
    text = re.sub('<[^>]*>','',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ''.join(emoticons).replace('-', '')
    tokenized = [porter.stem(word) for word in text.split() if word not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore', n_features = 2**21,
                         preprocessor=None, tokenizer=tokenizer)