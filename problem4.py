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


word_index_dict = {}
# Read brown_vocab_100.txt into word_index_dict
with open('brown_vocab_100.txt') as brown_file:
    word_index_dict={word.rstrip(): i for i, word in enumerate(brown_file)}

f = open("brown_100.txt")
dictionary= {i.lower(): j for j,i in enumerate(word_index_dict)}

# Initialize numpy 0s array
counts = np.zeros((len(word_index_dict),len(word_index_dict)))
counts+= 0.1

# Tokenise brown_100 words
words=[]
for line in f:
    text = line.split()
    for i in text:
        words.append(i.lower())

# Iterate through file and update counts
prev_word = '<s>'
for word in words[1:]:
    counts[word_index_dict[prev_word]][word_index_dict[word]] += 1
    prev_word=word

# Normalize counts
probs = normalize(counts, norm='l1', axis=1)

# Writeout smooth probabilities to smooth_probs.txt
np.savetxt("smooth_probs.txt", probs)

# Create function to easily access probabilities
# Bigram('the','all') gives the probability of 'all the' given 'all'
def bigram(word, prev_word):
    return probs[word_index_dict[prev_word]][word_index_dict[word]]

# Add probabilities to end of file
b = open("smooth_probs.txt", 'a')
b.write(str(bigram('the','all'))+'\n')
b.write(str(bigram('jury','the'))+'\n')
b.write(str(bigram('campaign','the'))+'\n')
b.write(str(bigram('calls','anonymous'))+'\n')

# Exercise 6 

t = open("toy_corpus.txt")

# Function to calculate the joint probability of a given sentence
def joint_prob_sent(sent):
    sentprob = 1
    sentance=sent.lower().split()
    for i in range(1,len(sentance)):
        sentprob *= bigram(sentance[i],sentance[i-1])
    return sentprob

# Function to calculate the perplexity of a given sentence
def perplexity(sent):
    return 1/pow(joint_prob_sent(sent), 1/(len(sent.split())-1))

# Add perplexities to file
b2= open("smoothed_eval.txt", 'w')
b2.write(str(perplexity(t.readline()))+'\n')
b2.write(str(perplexity(t.readline()))+'\n')

# Exercise 7 

b3 = open("smoothed_generation.txt",'w')

# Generate 10 sentences
for i in range(0,10):
    b3.write(GENERATE(word_index_dict, probs, 'bigram', max_words=25, start_word='<s>')+'\n')

# Close all open files
b3.close()
b2.close()
t.close()
b.close()
brown_file.close()
f.close()