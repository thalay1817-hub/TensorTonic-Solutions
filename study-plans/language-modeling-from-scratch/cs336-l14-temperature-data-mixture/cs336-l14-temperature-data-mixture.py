import numpy as np

def temperature_data_mixture(token_counts, alpha, training_budget):
    n = len(token_counts)
    counts = np.array(token_counts, dtype=np.float64)

    probabilities = np.zeros(n, dtype=np.float64)
    expected_tokens = np.zeros(n, dtype=np.float64)
    expected_epochs = np.zeros(n, dtype=np.float64)

    positive_mask = counts > 0

    if np.any(positive_mask):
        positive_counts = counts[positive_mask]

        # Compute log weights = alpha * log(n_i) for positive sources only
        log_weights = alpha * np.log(positive_counts)

        # Max-shift for numerical stability
        max_log_weight = np.max(log_weights)
        shifted = log_weights - max_log_weight
        weights = np.exp(shifted)

        total_weight = np.sum(weights)
        positive_probs = weights / total_weight

        probabilities[positive_mask] = positive_probs

        pos_expected_tokens = training_budget * positive_probs
        expected_tokens[positive_mask] = pos_expected_tokens

        pos_expected_epochs = training_budget * positive_probs / positive_counts
        expected_epochs[positive_mask] = pos_expected_epochs

    return {
        "probabilities": probabilities,
        "expected_tokens": expected_tokens,
        "expected_epochs": expected_epochs,
    }