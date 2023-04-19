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

#TODO: initialize numpy 0s array
counts = np.matrix(word_index_dict)
print(counts)

#TODO: iterate through file and update counts

#TODO: normalize counts

#TODO: writeout bigram probabilities to bigram_probs.txt
bigrams = open("bigram_probs.txt", 'w')


bigrams.close()
vocab.close()
f.close()