def train_bpe(corpus, vocab_size):
    # Convert each string to a list of UTF-8 byte ids
    sequences = [list(s.encode('utf-8')) for s in corpus]

    # Initial vocabulary: 256 single-byte tokens
    token_bytes = {i: bytes([i]) for i in range(256)}

    vocab_entries = []
    merges = []
    next_id = 256

    while next_id < vocab_size:
        # Count adjacent pairs within each sequence (never across sequences)
        pair_counts = {}
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

        if not pair_counts:
            break

        # Select most frequent pair; ties broken by lexicographically
        # greater pair of byte strings (token_bytes[left], token_bytes[right])
        best_pair = max(
            pair_counts.keys(),
            key=lambda p: (pair_counts[p], (token_bytes[p[0]], token_bytes[p[1]]))
        )

        left, right = best_pair
        new_bytes = token_bytes[left] + token_bytes[right]
        new_id = next_id

        token_bytes[new_id] = new_bytes
        vocab_entries.append([new_id, list(new_bytes)])
        merges.append([left, right, new_id])

        # Replace occurrences of (left, right) non-overlapping, left to right
        new_sequences = []
        for seq in sequences:
            new_seq = []
            i = 0
            n = len(seq)
            while i < n:
                if i < n - 1 and seq[i] == left and seq[i + 1] == right:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_sequences.append(new_seq)
        sequences = new_sequences

        next_id += 1

    return {"vocab": vocab_entries, "merges": merges}