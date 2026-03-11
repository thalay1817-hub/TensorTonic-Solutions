import numpy as np
import math
from collections import Counter

def tfidf_vectorizer(documents):
    N = len(documents)

    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in documents]

    # Build vocabulary
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))
    vocab_index = {word: i for i, word in enumerate(vocab)}

    # Document frequency
    df = {word: 0 for word in vocab}
    for doc in tokenized_docs:
        for word in set(doc):
            df[word] += 1

    # Compute IDF
    idf = {word: math.log(N / df[word]) for word in vocab}

    # Initialize TF-IDF matrix
    matrix = np.zeros((N, len(vocab)))

    # Fill TF-IDF values
    for i, doc in enumerate(tokenized_docs):
        counts = Counter(doc)
        total_terms = len(doc)

        for word, count in counts.items():
            j = vocab_index[word]
            tf = count / total_terms
            matrix[i][j] = tf * idf[word]

    return matrix, vocab


# Example 1
documents = ["the cat sat", "the dog ran"]
matrix, vocab = tfidf_vectorizer(documents)
print("matrix shape:", matrix.shape)
print("vocab:", vocab)

# Example 2
documents = ["apple banana", "banana cherry"]
matrix, vocab = tfidf_vectorizer(documents)
print("matrix shape:", matrix.shape)
print("vocab:", vocab)