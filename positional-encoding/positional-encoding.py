import numpy as np

def positional_encoding(seq_len, d_model, base=10000):
    # Positions (T, 1)
    pos = np.arange(seq_len)[:, np.newaxis]
    
    # Number of frequency pairs
    num_freqs = (d_model + 1) // 2
    
    # Frequencies
    i = np.arange(num_freqs)[np.newaxis, :]
    
    # Compute angle rates
    angle_rates = 1 / (base ** (2 * i / d_model))
    
    # Angles
    angles = pos * angle_rates
    
    # Initialize encoding matrix
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # Fill even indices (sin)
    pe[:, 0::2] = np.sin(angles[:, :pe[:, 0::2].shape[1]])
    
    # Fill odd indices (cos)
    pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])
    
    return pe   # <-- returns np.ndarray