import math
import torch

def densenet_channel_counts(stem_channels, growth_rate, block_layers, compression):
    counts = [stem_channels]
    c = stem_channels
    for i, n in enumerate(block_layers):
        c = c + n * growth_rate
        counts.append(c)
        if i < len(block_layers) - 1:
            c = math.floor(compression * c)
            counts.append(c)
    return torch.tensor(counts, dtype=torch.int64)