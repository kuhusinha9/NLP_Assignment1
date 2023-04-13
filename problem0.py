import nltk
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import brown
from nltk import word_tokenize
import itertools

# We chose lore and science_fiction
genres = ['lore', 'science_fiction']
# Import brown txt

# Dictionary with word count
counts={}
for word in brown.words():
    if word not in counts.keys():
        counts[word]=0
    counts[word]+=1
count = sorted(counts.items(), key=lambda item: item[1], reverse=True)
plt.plot([x[0] for x in count], [x[1] for x in count])


for genre in genres:
    words = brown.words(categories = genre)
    sentences = brown.sents(categories = genre)
    types = brown.tagged_words(categories = genre)
    
    # tokenize
    tokens = words

    #Number of tokens
    num_of_tokens = len(tokens)
    print(f'The number of tokens in the {genre} genre is: {num_of_tokens}')

    #Number of types
    unique_types = set([pair[1] for pair in types])
    num_of_types = len(unique_types)
    print(f'The number of unique types in the {genre} is: {num_of_types}')

    #Number of words
    

    #Average number of words per sentence

    #Average word length. 
    word_lengths = []
    for word in words:
        word_lengths.append(len(word))
    print(f'The average word length in the {genre} genre is: {np.mean(word_lengths)}')
    
    #Default part-of-speech tagger on the dataset and identify the ten most frequent POS tags.
    tags = brown.tagged_words()

    