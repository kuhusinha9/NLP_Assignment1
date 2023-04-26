#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""
import numpy as np
from generate import GENERATE
from problem1 import wordindex

def uniprobs(file: str, word_index_dict: dict)->np.ndarray:
    """_summary_

    Args:
        file (str): A filename of a file containing sentences (corpus).
        word_index_dict (dict): A dictionary mapping words to indexes.

    Returns:
        np.ndarray: Contains an array with unigram probabilities for each word in the corpus.
    """
    
    # Open file
    f = open(file)

    # Initialize counts to a zero vector
    counts=np.zeros(len(word_index_dict))

    # Iterate through file and update counts
    for sentence in f:
        for word in sentence.split():
            counts[word_index_dict[word.lower()]]+=1

    # Close file
    f.close()

    # Normalize and writeout counts 
    probs = counts / np.sum(counts)
    np.savetxt("probabilities/unigram_probs.txt", probs)
    
    return probs


if __name__ == "__main__":

    # Exercise 2

    # Open file
    vocab = open("assigned_txts/brown_vocab_100.txt")

    # Load the indices dictionary
    word_index_dict=wordindex(vocab)

    # Calculate uniform probabilities and save in unigram_probs.txt
    probs=uniprobs("assigned_txts/brown_100.txt", word_index_dict)

    # Exercise 6

    # Open toy corpus file
    f = open("assigned_txts/toy_corpus.txt")

    with open('evaluation/unigram_eval.txt', 'w') as uniprobs:
        # Iterate through sentences in the toy corpus
        for sentence in f:
            sent_len=len(sentence.split())
            sentprob = 1
            # Iterate through words 
            for word in sentence.split(): 
                # Calculate joint probability of all the words in the sentence
                sentprob *= probs[word_index_dict[word.lower()]]
            
            # Calculate perplexity and save to file
            perplexity = 1/(pow(sentprob, 1.0/sent_len))
            uniprobs.write(str(perplexity) + "\n")
    # Close toy corpus file
    uniprobs.close()

    # Exercise 7

    # Generate 10 sentences
    with open('generation/unigram_generation.txt', 'w') as f:
        for i in range(10):
            f.write(GENERATE(word_index_dict, probs, "unigram", max_words=25, start_word="<s>") + "\n")