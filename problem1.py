#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

word_index_dict = {}

# read brown_vocab_100.txt into word_index_dict
with open('brown_vocab_100.txt') as brown_file:
    word_index_dict={word.rstrip(): i for i, word in enumerate(brown_file)}
          
# write word_index_dict to word_to_index_100.txt

wf = open('word_to_index_100.txt','w')
wf.write(str(word_index_dict)) 
wf.close()


print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
