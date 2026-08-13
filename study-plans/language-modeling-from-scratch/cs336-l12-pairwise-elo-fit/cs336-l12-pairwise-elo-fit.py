import math
import torch

def pairwise_elo_fit(
    num_models,
    model_a,
    model_b,
    outcomes,
    steps,
    learning_rate,
):
    device = outcomes.device
    M = model_a.shape[0]

    model_a64 = model_a.to(torch.int64)
    model_b64 = model_b.to(torch.int64)
    outcomes64 = outcomes.to(torch.float64)

    ratings = torch.zeros(num_models, dtype=torch.float64, device=device)

    ln10_over_400 = math.log(10) / 400.0

    for _ in range(steps):
        ra = ratings[model_a64]
        rb = ratings[model_b64]
        pn = 1.0 / (1.0 + torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device), (rb - ra) / 400.0))
        residual = outcomes64 - pn

        coeff = ln10_over_400 / M if M > 0 else 0.0
        contrib = residual * coeff

        grad = torch.zeros(num_models, dtype=torch.float64, device=device)
        if M > 0:
            grad.index_add_(0, model_a64, contrib)
            grad.index_add_(0, model_b64, -contrib)

        ratings = ratings + learning_rate * grad
        ratings = ratings - ratings.mean()

    # Final fitted probabilities using final ratings
    if M > 0:
        ra_final = ratings[model_a64]
        rb_final = ratings[model_b64]
        fitted_probabilities = 1.0 / (1.0 + torch.pow(torch.tensor(10.0, dtype=torch.float64, device=device), (rb_final - ra_final) / 400.0))
    else:
        fitted_probabilities = torch.zeros(0, dtype=torch.float64, device=device)

    # Rankings: descending rating, ties broken by lower index
    indices = torch.arange(num_models, dtype=torch.int64, device=device)
    # sort by (-rating, index) lexicographically
    neg_ratings = -ratings
    # stable sort by index first, then by neg_ratings (stable sort preserves index order for ties)
    order_by_index = torch.argsort(indices, stable=True)
    sorted_by_rating = torch.argsort(neg_ratings[order_by_index], stable=True)
    rankings = order_by_index[sorted_by_rating]
    rankings = rankings.to(torch.int64)

    # Union-Find for connected components
    parent = list(range(num_models))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry

    model_a_list = model_a64.tolist()
    model_b_list = model_b64.tolist()
    for a, b in zip(model_a_list, model_b_list):
        union(a, b)

    roots = [find(i) for i in range(num_models)]
    root_to_id = {}
    component_ids_list = []
    next_id = 0
    for r in roots:
        if r not in root_to_id:
            root_to_id[r] = next_id
            next_id += 1
        component_ids_list.append(root_to_id[r])

    component_ids = torch.tensor(component_ids_list, dtype=torch.int64, device=device)
    connected = bool(next_id == 1)

    return {
        "ratings": ratings,
        "rankings": rankings,
        "fitted_probabilities": fitted_probabilities,
        "connected": connected,
        "component_ids": component_ids,
    }