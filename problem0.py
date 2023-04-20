import nltk, re, itertools
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import brown
from nltk import word_tokenize

# We chose lore and science_fiction
genres = ['lore', 'government', None]
genres_names = ['Lore genre', 'Science Fiction genre', 'all genres']
# Import brown txt

# print brown categories
print(brown.categories())

#belles letres text store in txt file
belles_lettres = list(brown.words(categories = 'belles_lettres'))
belles_letres_txt = open('belles_lettres.txt', 'w')
belles_letres_txt.write(str(belles_lettres))
belles_letres_txt.close()

# store the pos tags in a txt file
belles_letres_pos = list(brown.tagged_words(categories = 'belles_lettres'))
belles_letres_pos_txt = open('belles_letres_pos.txt', 'w')
belles_letres_pos_txt.write(str(belles_letres_pos))
belles_letres_pos_txt.close()

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
    
    pos_tag_first = pos_tags[0]
    count = 1
    pos_tag_count_dict = {}
    for pos_tag in pos_tags[1:]:
        if pos_tag == pos_tag_first:
            count += 1
        else:
            pos_tag_count_dict[pos_tag_first] = count
            pos_tag_first = pos_tag
            count = 1

    pos_tags = sorted(pos_tag_count_dict.items(), key=lambda item: item[1], reverse=True)
    pos_tags_names = [pair[0] for pair in pos_tags]
    print(f'The ten most frequent POS tags in the {genres_names[idx]} genre are: {pos_tags[:10]}')
    

    
