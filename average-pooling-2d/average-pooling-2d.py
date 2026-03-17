def average_pooling_2d(X, pool_size):
    H = len(X)
    W = len(X[0])
    
    # Output dimensions
    out_h = H // pool_size
    out_w = W // pool_size
    
    # Initialize output
    output = [[0.0 for _ in range(out_w)] for _ in range(out_h)]
    
    # Perform pooling
    for i in range(out_h):
        for j in range(out_w):
            total = 0.0
            
            # Sum over pool window
            for a in range(pool_size):
                for b in range(pool_size):
                    total += X[i * pool_size + a][j * pool_size + b]
            
            # Compute average
            output[i][j] = total / (pool_size * pool_size)
    
    return output


# Example usage
X1 = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

X2 = [
    [10, 20],
    [30, 40]
]

print(average_pooling_2d(X1, 2))
print(average_pooling_2d(X2, 2))