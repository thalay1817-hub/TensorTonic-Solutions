import numpy as np

def silhouette_score(X, labels):
    X = np.array(X, dtype=float)
    labels = np.array(labels)
    n = len(X)
    
    # Compute pairwise Euclidean distance matrix
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    unique_labels = np.unique(labels)
    silhouettes = []
    
    for i in range(n):
        same_cluster = labels == labels[i]
        other_clusters = labels != labels[i]
        
        # a(i): mean intra-cluster distance (exclude self)
        same_idx = np.where(same_cluster)[0]
        if len(same_idx) > 1:
            a = np.mean([dist_matrix[i, j] for j in same_idx if j != i])
        else:
            a = 0.0
        
        # b(i): minimum mean distance to other clusters
        b = float('inf')
        for label in unique_labels:
            if label == labels[i]:
                continue
            cluster_idx = np.where(labels == label)[0]
            b = min(b, np.mean(dist_matrix[i, cluster_idx]))
        
        # silhouette score for point i
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouettes.append(s)
    
    return float(np.mean(silhouettes))