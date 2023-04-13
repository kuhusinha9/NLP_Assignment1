import nltk, matplotlib
from nltk.corpus import brown
from nltk import word_tokenize
import itertools
print(brown.categories())
# We chose lore and science_fiction
genres = ['lore', 'science_fiction']
# Import brown txt


for genre in genres:
    # raw = brown.raw(categories = genre)
    # words = brown.words(categories = genre)
    
    # # tokenize
    # print(raw[:100])
    # tokens = word_tokenize(raw)
    # print(len(words))
    # print(len(tokens))
    # print(words[:10])
    # print(tokens[:10])


    words = brown.words(categories = genre)
    types = brown.tagged_words(categories = genre)
    # tokenize
    tokens = words

    #Number of tokens
    num_of_tokens = len(tokens)
    print(f'The number of tokens in the {genre} genre is: {num_of_tokens}')

    #Number of types
    print(types[:10])
    types = [pair[1] for pair in types]
    unique_types = set(types)
    num_of_types = len(unique_types)
    print(num_of_types)
    #Number of tokens
    # print("," in words)
    # num_of_tokens = len(words)
    # print(f'The number of tokens in the {genre} genre is: {num_of_tokens}')
    #Number of types

    #Number of words

    #Average number of words per sentence

    #Average word length. 


    #Default part-of-speech tagger on the dataset and identify the ten most frequent POS tags.