def remove_stopwords(tokens, stopwords):
    # convert stopwords list to set for fast lookup
    stop_set = set(stopwords)
    
    # keep only tokens not in stopwords
    result = [token for token in tokens if token not in stop_set]
    
    return result


# Example 1
tokens = ["this", "is", "a", "test"]
stopwords = ["is", "a"]
print(remove_stopwords(tokens, stopwords))

# Example 2
tokens = ["hello", "world"]
stopwords = ["the", "and"]
print(remove_stopwords(tokens, stopwords))

# Example 3
tokens = ["a", "an", "the"]
stopwords = ["a", "an", "the"]
print(remove_stopwords(tokens, stopwords))