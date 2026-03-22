import numpy as np

def replay_buffer_sample(buffer, batch_size, seed=None):
    rng = np.random.RandomState(seed)
    
    # Sample indices without replacement
    indices = rng.choice(len(buffer), size=batch_size, replace=False)
    
    # Return sampled transitions
    return [buffer[i] for i in indices]