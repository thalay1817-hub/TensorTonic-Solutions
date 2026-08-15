def cache_aware_prefill_routing(request_token_counts, cached_prefix_lengths, worker_ids, worker_capacities):
    num_workers = len(worker_ids)

    assigned_tokens = [0] * num_workers
    uncached_loads = [0] * num_workers

    assignments = []

    for req_idx, token_count in enumerate(request_token_counts):
        prefixes = cached_prefix_lengths[req_idx]

        candidates = []
        for w_idx in range(num_workers):
            capacity = worker_capacities[w_idx]
            projected_assigned = assigned_tokens[w_idx] + token_count

            if projected_assigned > capacity:
                continue  # infeasible

            cached_prefix = prefixes[w_idx]
            uncached_suffix = token_count - cached_prefix
            projected_uncached_load = uncached_loads[w_idx] + uncached_suffix
            remaining_capacity = capacity - projected_assigned
            worker_id = worker_ids[w_idx]

            # Sort key: longest cached prefix (desc), lowest projected uncached load (asc),
            # greatest remaining capacity (desc), lowest worker id (asc)
            sort_key = (-cached_prefix, projected_uncached_load, -remaining_capacity, worker_id)

            candidates.append((sort_key, w_idx, worker_id, uncached_suffix))

        if not candidates:
            assignments.append(-1)
            continue

        candidates.sort(key=lambda c: c[0])
        _, chosen_w_idx, chosen_worker_id, chosen_uncached_suffix = candidates[0]

        assigned_tokens[chosen_w_idx] += token_count
        uncached_loads[chosen_w_idx] += chosen_uncached_suffix

        assignments.append(chosen_worker_id)

    return {
        "assignments": assignments,
        "worker_uncached_loads": uncached_loads,
    }