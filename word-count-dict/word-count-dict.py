def word_count_dict(sentences):
    freq = {}

    for sentence in sentences:
        for word in sentence:
            freq[word] = freq.get(word, 0) + 1

    return freq


# Example 1
sentences = [["i", "love", "ml"], ["i", "love", "coding"]]
print(word_count_dict(sentences))

# Example 2
sentences = [["hello", "hello"], ["world"]]
print(word_count_dict(sentences))

# Example 3
sentences = []
print(word_count_dict(sentences))