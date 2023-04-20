#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""
import numpy as np
from generate import GENERATE

vocab = open("brown_vocab_100.txt")

# load the indices dictionary
word_index_dict={line.rstrip(): i for i, line in enumerate(vocab)}

# open file
f = open("brown_100.txt")

# initialize counts to a zero vector
counts=np.zeros(len(word_index_dict))

# iterate through file and update counts
for sentence in f:
    words=[word.lower()for word in sentence.split() if word != "<s>" and word != "</s>"] 
    for word in sentence.split():
        counts[word_index_dict[word.lower()]]+=1

proportion= np.count_nonzero(counts == 1)/len(word_index_dict)

# close file
f.close()

# normalize and writeout counts. 
probs = counts / np.sum(counts)
np.savetxt("unigram_probs.txt", probs)