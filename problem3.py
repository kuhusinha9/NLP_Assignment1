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
from problem1 import word_index_dict


word_index_dict = {}
# read brown_vocab_100.txt into word_index_dict
with open('brown_vocab_100.txt') as brown_file:
    word_index_dict={word.rstrip(): i for i, word in enumerate(brown_file)}

f = open("brown_100.txt")
dictionary= {i.lower(): j for j,i in enumerate(word_index_dict)}

#initialize numpy 0s array
counts = np.zeros((len(word_index_dict),len(word_index_dict)))

# tokenise brown_100 words
words=[]
for line in f:
    text = line.split()
    for i in text:
        words.append(i.lower())

#iterate through file and update counts
prev_word = '<s>'
for word in words[1:]:
    counts[word_index_dict[prev_word]][word_index_dict[word]] += 1
    prev_word=word

#normalize counts
probs = normalize(counts, norm='l1', axis=1)

#writeout bigram probabilities to bigram_probs.txt
np.savetxt("bigram_probs.txt", probs)

# create function to easily access probabilities
# bigram('the','all') gives the probability of 'all the' given 'all'
def bigram(word, prev_word):
    return probs[word_index_dict[prev_word]][word_index_dict[word]]

# add probabilities to end of file
b = open("bigram_probs.txt", 'a')
b.write(str(bigram('the','all'))+'\n')
b.write(str(bigram('jury','the'))+'\n')
b.write(str(bigram('campaign','the'))+'\n')
b.write(str(bigram('calls','anonymous'))+'\n')

#
# Task 6 code
#
t = open("toy_corpus.txt")

# function to calculate the joint probability of a given sentence
def joint_prob_sent(sent):
    sentprob = 1
    sentance=sent.lower().split()
    for i in range(1,len(sentance)):
        sentprob *= bigram(sentance[i],sentance[i-1])
    return sentprob

# function to calculate the perplexity of a given sentence
def perplexity(sent):
    return 1/pow(joint_prob_sent(sent), 1/(len(sent.split())-1))

# add perplexities to file
b2= open("bigram_eval.txt", 'w')
b2.write(str(perplexity(t.readline()))+'\n')
b2.write(str(perplexity(t.readline()))+'\n')

#
# Task 7 code
#
b3 = open("bigram_generation.txt",'w')

#generate 10 sentences
for i in range(0,10):
    b3.write(GENERATE(word_index_dict, probs, 'bigram', max_words=25, start_word='<s>')+'\n')

# close all open files
b3.close()
b2.close()
t.close()
b.close()
brown_file.close()
f.close()