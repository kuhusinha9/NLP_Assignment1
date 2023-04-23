import numpy as np
import nltk
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import brown
from nltk import word_tokenize
import itertools
import copy

# Pmi formula
sums= lambda x: sum(x.values()) # takes sum of dict values
pmi = lambda x, N, y, z: np.log((x * N) / (sums(y) * sums(z)))

# Word dict
word_index_dict = {}

# Read words
tokens = brown.words()

last_word = tokens[0]

for idx, token in enumerate(tokens):
    # strip token
    token = token.rstrip().lower()
    if last_word not in word_index_dict:
        word_index_dict[last_word] = {}
    if token not in word_index_dict[last_word]:
        word_index_dict[last_word][token] = 0
    
    word_index_dict[last_word][token] += 1
    last_word = token


# Remove words that occur less than 10 times 
word_index_copy = copy.deepcopy(word_index_dict)

for word in word_index_copy:
    occurences = sums(word_index_copy[word])
    if occurences < 10:
        del word_index_dict[word]

pmi_list = []
# Calculate pmi
for word in word_index_dict:
    for token in word_index_dict[word]:
        if token in word_index_dict: # I am not sure what to do when the token is not in the dict
            pmi_value = pmi(word_index_dict[word][token], len(tokens), (word_index_dict[word]), word_index_dict[token])
            pmi_list.append((word, token, round(pmi_value,2)))

# Sort by pmi
pmi_list = sorted(pmi_list, key=lambda item: item[2], reverse=True)

# Print top 20 
print("The top 20")
for i in range(20):
    print(i+1, pmi_list[i])

# Print last 20
print("The last 20")
for i in range(1,21):
    print(i, pmi_list[-i])

    