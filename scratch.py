import nltk
from nltk.tokenize import word_tokenize


# Load the corpus
brown_corpus = nltk.corpus.brown

# Get the tokens
tokens = word_tokenize(" ".join(brown_corpus.words()))

# Get the number of tokens
num_tokens = len(tokens)

print("Number of tokens in the Brown Corpus:", num_tokens)
