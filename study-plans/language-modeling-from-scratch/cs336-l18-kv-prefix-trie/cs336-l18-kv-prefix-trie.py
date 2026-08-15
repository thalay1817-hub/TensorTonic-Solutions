def kv_prefix_trie(cached_sequences, requests):
    # Node 0 is the root. Each node has a dict mapping token -> child node id.
    children = [{}]  # children[node_id] = {token: child_node_id}
    next_node_id = 1

    for sequence in cached_sequences:
        current = 0
        for token in sequence:
            if token in children[current]:
                current = children[current][token]
            else:
                new_id = next_node_id
                next_node_id += 1
                children[current][token] = new_id
                children.append({})
                current = new_id

    matches = []
    for request in requests:
        current = 0
        matched_length = 0
        for token in request:
            if token in children[current]:
                current = children[current][token]
                matched_length += 1
            else:
                break

        uncached_suffix = list(request[matched_length:])

        matches.append({
            "matched_length": matched_length,
            "cache_node": current,
            "uncached_suffix": uncached_suffix,
        })

    return {"matches": matches}