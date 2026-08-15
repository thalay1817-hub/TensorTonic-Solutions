import json
import hashlib
import numpy as np

def minhash_signatures(documents, shingle_size, num_hashes, seed):
    EMPTY_VAL = (1 << 32) - 1

    num_docs = len(documents)
    signatures = np.empty((num_docs, num_hashes), dtype=np.uint32)
    empty_flags = [False] * num_docs

    for doc_idx, tokens in enumerate(documents):
        # Build shingle set (deduplicated) as compact JSON strings
        shingle_strs = set()
        n = len(tokens)
        for start in range(n - shingle_size + 1):
            window = tokens[start:start + shingle_size]
            encoded = json.dumps(window, separators=(",", ":"), ensure_ascii=False)
            shingle_strs.add(encoded)

        if not shingle_strs:
            empty_flags[doc_idx] = True
            signatures[doc_idx, :] = EMPTY_VAL
            continue

        # Precompute UTF-8 bytes for each shingle once
        shingle_bytes_list = [s.encode("utf-8") for s in shingle_strs]

        for h in range(num_hashes):
            prefix = f"{seed}:{h}:".encode("utf-8")
            min_val = EMPTY_VAL
            for shingle_bytes in shingle_bytes_list:
                digest = hashlib.sha256(prefix + shingle_bytes).digest()
                val = int.from_bytes(digest[:4], byteorder="big", signed=False)
                if val < min_val:
                    min_val = val
            signatures[doc_idx, h] = min_val

    # Compute similarities
    similarities = np.empty((num_docs, num_docs), dtype=np.float64)
    for i in range(num_docs):
        for j in range(num_docs):
            if i == j:
                similarities[i, j] = 1.0
                continue
            if empty_flags[i] and empty_flags[j]:
                similarities[i, j] = 1.0
            elif empty_flags[i] != empty_flags[j]:
                similarities[i, j] = 0.0
            else:
                matches = np.sum(signatures[i] == signatures[j])
                similarities[i, j] = matches / num_hashes

    return {
        "signatures": signatures,
        "similarities": similarities,
    }