def detect_drift(reference_counts, production_counts, threshold):
    # Step 1: Normalize histograms
    ref_total = sum(reference_counts)
    prod_total = sum(production_counts)

    ref = [x / ref_total for x in reference_counts]
    prod = [x / prod_total for x in production_counts]

    # Step 2: Compute Total Variation Distance
    diff_sum = sum(abs(p - q) for p, q in zip(ref, prod))
    tvd = 0.5 * diff_sum

    # Step 3: Drift detection (strictly > threshold)
    drift_detected = tvd > threshold

    return {"score": tvd, "drift_detected": drift_detected}