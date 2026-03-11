def cohens_kappa(rater1, rater2):
    n = len(rater1)

    # Observed agreement
    agreements = sum(1 for a, b in zip(rater1, rater2) if a == b)
    p_o = agreements / n

    # Collect distinct labels
    labels = set(rater1) | set(rater2)

    # Expected agreement
    p_e = 0
    for label in labels:
        p1 = rater1.count(label) / n
        p2 = rater2.count(label) / n
        p_e += p1 * p2

    # Handle denominator zero case
    if p_e == 1:
        return 1.0

    kappa = (p_o - p_e) / (1 - p_e)
    return float(kappa)


# Example 1
rater1 = [0, 1, 0, 1]
rater2 = [0, 1, 0, 1]
print(cohens_kappa(rater1, rater2))

# Example 2
rater1 = [0, 0, 1, 1]
rater2 = [0, 1, 1, 0]
print(cohens_kappa(rater1, rater2))