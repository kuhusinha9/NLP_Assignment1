import nltk, re, itertools
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import brown
from nltk import word_tokenize


def most_freq_pos_tag(words):
    """
    Takes the words and determines the pos tags. Then it returns the ordered pos tags based on frequency
    """
    pos_tags = nltk.pos_tag(words) # default NLTK dagger
    pos_tags = [pair[1] for pair in pos_tags]
    pos_tags = sorted(pos_tags) # sort all the pos tags
    
    pos_tag_first = pos_tags[0]
    count = 1
    pos_tag_count_dict = {} # dictionary to store the pos tag and its count
    for pos_tag in pos_tags[1:]:
        if pos_tag == pos_tag_first: 
            count += 1
        else:
            pos_tag_count_dict[pos_tag_first] = count
            pos_tag_first = pos_tag
            count = 1

    pos_tags = sorted(pos_tag_count_dict.items(), key=lambda item: item[1], reverse=True) # pos_tags with counts
    pos_tags_names = [pair[0] for pair in pos_tags] # pos_tags names only

    # Add ranks to the word frequency
    for idx, tag in enumerate(pos_tags):
        pos_tags[idx] = (tag[0], tag[1], f'({idx+1})')
    return pos_tags

def get_word_frequecy(words):
    """
    Calculates the word frequency for the words
    """
    word_frequency = {}
    for word in words:
        word = word.rstrip()
        word = word.lower()
        if word not in word_frequency.keys():
            word_frequency[word]=0
        word_frequency[word]+=1

    word_frequency = sorted(word_frequency.items(), key=lambda item: item[1], reverse=True)

    return word_frequency

def plot_zipf(word_frequency, genres_names, idx):
    # Get max frequency for x axis
    max_freq = len(word_frequency)
    # Round on nearest 1000
    rounder1 = lambda x: int(x/1000)*1000
    max_freq = rounder1(max_freq)
    
    # Plot Zipf's law for frequencies, two supplots one figure
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in word_frequency], [x[1] for x in word_frequency])
    plt.title(f'Zipf\'s law for frequencies in {genres_names[idx]}')
    plt.xlabel('Rank')
    # Set x labels less frequent set 5 ticks from 0 to max_freq
    plt.xticks([0, max_freq/4, max_freq/2, 3*max_freq/4, max_freq], [0, int(max_freq/4), int(max_freq/2), int(3*max_freq/4), max_freq])
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    # Log-log plot
    plt.loglog([x[0] for x in word_frequency], [x[1] for x in word_frequency])
    plt.title(f'Zipf\'s law for frequencies in {genres_names[idx]}')
    # Set x labels log scale 5 ticks from 1 to max_freq log scale
    plt.xticks([int(max_freq**0.2), int(max_freq**0.4), int(max_freq**0.6), int(max_freq**0.8), max_freq], [int(max_freq**0.2), int(max_freq**0.4), int(max_freq**0.6), int(max_freq**0.8), max_freq])

    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    # Save fig in plots folder
    plt.savefig(f'plots/Zipf\'s law for frequencies in {genres_names[idx]}.png')
    plt.close()

def get_raw_text(sentences):
    # concatenate words per sentence
    raw_text = []
    for sentence in sentences:
        raw_text.append(' '.join(sentence))
    
    # concatenate sentences with new sentence token
    raw_text = ' '.join(raw_text)

    return raw_text

def main(genres, genres_names):

    # Dictionary with word count
    counts={}
    for word in brown.words():
        if word not in counts.keys():
            counts[word]=0
        counts[word]+=1
    count = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    plt.plot([x[0] for x in count], [x[1] for x in count])


    for idx, genre in enumerate(genres):
        words = brown.words(categories = genre)
        sentences = brown.sents(categories = genre)
        types = brown.tagged_words(categories = genre)
        raw_text = get_raw_text(sentences)
        
        #Number of tokens
        tokens = word_tokenize(raw_text)
        num_of_tokens = len(tokens)
        print(f'The number of tokens in the {genres_names[idx]} is: {num_of_tokens}')

        # Number of types
        unique_types = set([pair[1] for pair in types])
        num_of_types = len(unique_types)
        print(f'The number of unique types in the {genres_names[idx]} is: {num_of_types}')

        # Number of words
        num_of_words = len(words)
        print(f'The number of words in the {genres_names[idx]} is: {num_of_words}')

        # Average number of words per sentence
        num_of_sentences = len(sentences)
        print(f'The average number of words per sentence in the {genres_names[idx]} is: {round(num_of_words/num_of_sentences,3)}')

        # Average word length. 
        word_lengths = []
        for word in words:
            word_lengths.append(len(word))
        print(f'The average word length in the {genres_names[idx]} genre is: {round(np.mean(word_lengths),3)}')
        
        # Run part-of-speech tagger on the dataset and identify the ten most frequent POS tags.
        pos_tags = most_freq_pos_tag(words)
        print(f'The ten most frequent POS tags in the {genres_names[idx]} genre are: {pos_tags[:18]}')

        # Get word counts of unique words
        word_frequency = get_word_frequecy(words)

        # Plot Zipf's law
        plot_zipf(word_frequency, genres_names, idx)

if __name__ == '__main__':
    # We chose lore and science_fiction
    genres = ['lore', 'government', None]
    genres_names = ['Lore genre', 'Science Fiction genre', 'all genres']

    main(genres, genres_names)