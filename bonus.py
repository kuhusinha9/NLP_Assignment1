import numpy as np
import nltk
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import brown
from nltk import word_tokenize
import itertools
import copy

# pmi formula
pmi = lambda x, N, y, z: np.log((x * N) / (y * z))

# word dict
word_index_dict = {}

# read words
tokens = brown.words()

last_word = tokens[0]

for token in tokens[1:1000]:
    # strip token
    token = token.rstrip().lower()
    if last_word not in word_index_dict:
        word_index_dict[last_word] = {}
    if token not in word_index_dict[last_word]:
        word_index_dict[last_word][token] = 0
        
    word_index_dict[last_word][token] += 1
    last_word = token

# remove words that occur less than 10 times 
word_index_copy = copy.deepcopy(word_index_dict)

for word in word_index_copy:
    occurences = 0
    for token in word_index_copy[word]:
        occurences += word_index_copy[word][token]
    if occurences < 10:
        del word_index_dict[word]

# calculate pmi
for word in word_index_dict:
    for token in word_index_dict[word]:
        if token in word_index_dict: # I am not sure what to do when the token is not in the dict
            word_index_dict[word][token] = pmi(word_index_dict[word][token], len(tokens), len(word_index_dict[word]), len(word_index_dict[token]))

print(word_index_dict)
# # sort by pmi
# for word in word_index_dict:
#     word_index_dict[word] = sorted(word_index_dict[word].items(), key=lambda item: item[1], reverse=True)

# # print top 20 and bottom 20
# for word in word_index_dict:
#     print(word)
#     print(word_index_dict[word][:20])
#     print(word_index_dict[word][-20:])
    