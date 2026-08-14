import torch
import math

def kv_cache_decode_step(query, new_key, new_value, key_cache, value_cache, num_query_heads, num_kv_heads):
    orig_dtype = query.dtype
    device = query.device

    B, Hq, D = query.shape
    Hkv = num_kv_heads
    G = num_query_heads // num_kv_heads

    # Append new_key/new_value to caches along sequence dim (S)
    new_key_cache = torch.cat([key_cache, new_key.unsqueeze(2)], dim=2)
    new_value_cache = torch.cat([value_cache, new_value.unsqueeze(2)], dim=2)

    # Compute attention in float32
    q = query.detach().to(torch.float32).reshape(B, Hkv, G, D)
    k = new_key_cache.detach().to(torch.float32)   # (B, Hkv, S+1, D)
    v = new_value_cache.detach().to(torch.float32)  # (B, Hkv, S+1, D)

    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum('bhgd,bhsd->bhgs', q, k) * scale  # (B, Hkv, G, S+1)

    attn = torch.softmax(scores, dim=-1)

    out = torch.einsum('bhgs,bhsd->bhgd', attn, v)  # (B, Hkv, G, D)

    output = out.reshape(B, Hq, D).to(orig_dtype).to(device)

    new_key_cache = new_key_cache.to(orig_dtype).to(device)
    new_value_cache = new_value_cache.to(orig_dtype).to(device)

    return {
        "output": output,
        "new_key_cache": new_key_cache,
        "new_value_cache": new_value_cache,
    }