def binning(values, num_bins):
    if not values:
        return []

    min_val = min(values)
    max_val = max(values)

    # If all values are equal
    if min_val == max_val:
        return [0] * len(values)

    # Compute bin width
    bin_width = (max_val - min_val) / num_bins

    bins = []
    for v in values:
        bin_idx = int((v - min_val) / bin_width)
        bin_idx = min(bin_idx, num_bins - 1)  # clamp to last bin
        bins.append(bin_idx)

    return bins


# Example 1
values = [0, 25, 50, 75, 100]
num_bins = 4
print(binning(values, num_bins))

# Example 2
values = [1, 2, 3, 4, 5, 6]
num_bins = 2
print(binning(values, num_bins))