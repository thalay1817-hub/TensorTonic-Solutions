def k_means_centroid_update(points, assignments, k):
    dim = len(points[0])

    # Initialize sums and counts
    centroids = [[0.0] * dim for _ in range(k)]
    counts = [0] * k

    # Sum points for each cluster
    for p, cluster in zip(points, assignments):
        counts[cluster] += 1
        for d in range(dim):
            centroids[cluster][d] += p[d]

    # Divide by counts to get mean
    for j in range(k):
        if counts[j] > 0:
            for d in range(dim):
                centroids[j][d] /= counts[j]

    return centroids


# Example 1
points = [[0, 0], [2, 2], [10, 10], [12, 12]]
assignments = [0, 0, 1, 1]
k = 2

print(k_means_centroid_update(points, assignments, k))


# Example 2
points = [[0, 0], [1, 0], [5, 5], [6, 5], [10, 0]]
assignments = [0, 0, 1, 1, 2]
k = 3

print(k_means_centroid_update(points, assignments, k))