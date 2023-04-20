#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
from problem1 import word_index_dict


vocab = open("brown_vocab_100.txt")
f = open("brown_100.txt")

#initialize numpy 0s array
counts = np.zeros((len(word_index_dict),len(word_index_dict)))

# tokenise brown_100 words
words=[]
for line in f:
    text = line.split()
    for i in text:
        words.append(i.lower())


#TODO: iterate through file and update counts
prev_word = '<s>'
for word in words:
    counts[word_index_dict[word]][word_index_dict[prev_word]] += 1
    prev_word=word

#normalize counts
probs = normalize(counts, norm='l1', axis=1)

#writeout bigram probabilities to bigram_probs.txt

np.savetxt("bigram_probs.txt", probs)

vocab.close()
f.close()