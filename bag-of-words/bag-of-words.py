import numpy as np

def bag_of_words_vector(tokens, vocab):
    # Create vocab -> index mapping
    vocab_index = {word: i for i, word in enumerate(vocab)}

    # Initialize count vector
    bow = np.zeros(len(vocab), dtype=int)

    # Count tokens
    for token in tokens:
        if token in vocab_index:
            bow[vocab_index[token]] += 1

    return bow


# Example 1
tokens = ["i", "love", "ml", "love"]
vocab = ["i", "love", "hate", "ml"]
print(bag_of_words_vector(tokens, vocab))

# Example 2
tokens = ["hello", "world"]
vocab = ["hello", "ml"]
print(bag_of_words_vector(tokens, vocab))

# Example 3
tokens = []
vocab = ["hello", "world"]
print(bag_of_words_vector(tokens, vocab))