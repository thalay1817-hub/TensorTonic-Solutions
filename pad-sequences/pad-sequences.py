import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    # Handle empty input
    if len(seqs) == 0:
        return np.array([])
    
    # Auto-detect max_len
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    # Initialize result with pad_value
    result = np.full((len(seqs), max_len), pad_value)
    
    # Copy sequences
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)   # handle truncation
        result[i, :length] = seq[:length]
    
    return result


# Example 1
seqs = [[1,2,3], [4,5], [6]]
print(pad_sequences(seqs, pad_value=0))

# Example 2
seqs = [[1,2,3,4], [5,6]]
print(pad_sequences(seqs, pad_value=-1, max_len=3))