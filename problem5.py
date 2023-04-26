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
import codecs
import json
from sklearn.preprocessing import normalize

vocab = codecs.open("assigned_txts/brown_vocab_100.txt")

# Read brown_100.txt into tokens
with open('assigned_txts/brown_100.txt') as brown_file:
    tokens = brown_file.read().split()

# Read brown_vocab_100.json into word_index_dict
with open('assigned_txts/word_to_index_100.json') as brown_file:
    word_index_dict = json.load(brown_file)

#p(past | in, the) (should be 0.0625 for unsmoothed, and ~0.011305 for smoothed)
#p(time | in, the)
#p(said | the, jury)
#p(recommended | the, jury)
#p(that | jury, said)
#p(, | agriculture, teacher

smoother = 0.1

# Initialize probability matrix length index dict as rows and 4 columns
prob_matrix = np.zeros((len(word_index_dict), 4)) + smoother
conditions = [('in', 'the'), ('the', 'jury'), ('jury', 'said'), ('agriculture', 'teacher')]
variables = [('past',("in",'the')), ('time',("in",'the')), ('said',("the",'jury')), ('recommended',("the",'jury')), ('that',("jury",'said')), (',',("agriculture",'teacher'))]
# Iterate through tokens
second_last_token = None
last_token = None
for token in tokens:
    token = token.rstrip().lower()
    last_two_tuple = (second_last_token, last_token)
    if last_two_tuple in conditions:
        prob_matrix[word_index_dict[token]][conditions.index(last_two_tuple)] += 1
    second_last_token = last_token
    last_token = token

# Normalize the matrix
prob_matrix = normalize(prob_matrix, axis=0, norm='l1')

# Print the probabilities
for variable in variables:
    print("p({} | {}, {})".format(variable[0], variable[1][0], variable[1][1]))
    print(round(prob_matrix[word_index_dict[variable[0]]][conditions.index(variable[1])],6))