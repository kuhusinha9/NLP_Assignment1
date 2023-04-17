import nltk, re, itertools
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import brown
from nltk import word_tokenize

# We chose lore and science_fiction
genres = ['lore', 'science_fiction', None]
genres_names = ['Lore genre', 'Science Fiction genre', 'all genres']
# Import brown txt

# Dictionary with word count
counts={}
for word in brown.words():
    if word not in counts.keys():
        counts[word]=0
    counts[word]+=1
count = sorted(counts.items(), key=lambda item: item[1], reverse=True)
plt.plot([x[0] for x in count], [x[1] for x in count])


for idx, genre in enumerate(genres):
    tokens = brown.words(categories = genre)
    sentences = brown.sents(categories = genre)
    types = brown.tagged_words(categories = genre)
    
    #Number of tokens
    num_of_tokens = len(tokens)
    print(f'The number of tokens in the {genres_names[idx]} is: {num_of_tokens}')

    #Number of types
    unique_types = set([pair[1] for pair in types])
    num_of_types = len(unique_types)
    print(f'The number of unique types in the {genres_names[idx]} is: {num_of_types}')

    #Number of words
    words = [word for word in tokens if word.isalpha()]
    num_of_words = len(words)
    print(f'The number of words in the {genres_names[idx]} is: {num_of_words}')

    #Average number of words per sentence
    num_of_sentences = len(sentences)
    print(f'The average number of words per sentence in the {genres_names[idx]} is: {round(num_of_words/num_of_sentences,3)}')

    #Average word length. 
    word_lengths = []
    for word in words:
        word_lengths.append(len(word))
    print(f'The average word length in the {genres_names[idx]} genre is: {round(np.mean(word_lengths),3)}')
    
    #Run part-of-speech tagger on the dataset and identify the ten most frequent POS tags.
    pos_tags = nltk.pos_tag(words)
    pos_tags = [pair[1] for pair in pos_tags]
    pos_tags = sorted(pos_tags)
    pos_tags = list(pos_tags for pos_tags,_ in itertools.groupby(pos_tags))
    pos_tags = sorted(pos_tags, key = lambda x: pos_tags.count(x), reverse = True)
    print(f'The ten most frequent POS tags in the {genres_names[idx]} genre are: {pos_tags[:10]}')
    

    
