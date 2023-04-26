#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

def wordindex(file:str)->dict:
    """ Returns word index dictionary

    Args:
        file (str): txt file containing all words

    Returns:
        dict: dictionary with words (keys) and their index (values)
    """
    word_index={word.rstrip(): i for i, word in enumerate(file)}
    return word_index

if __name__ == "__main__":

    word_index_dict = {}

    # Read brown_vocab_100.txt into word_index_dict
    with open('assigned_txts/brown_vocab_100.txt') as brown_file:
        word_index_dict=wordindex(brown_file)
            
    # Write word_index_dict to word_to_index_100.txt
    wf = open('assigned_txts/word_to_index_100.txt','w')
    wf.write(str(word_index_dict)) 
    wf.close()

    print(word_index_dict['all'])
    print(word_index_dict['resolution'])
    print(len(word_index_dict))