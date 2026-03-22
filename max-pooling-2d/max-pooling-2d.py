def max_pooling_2d(X, pool_size):
    H = len(X)
    W = len(X[0])
    
    # Output dimensions
    out_h = H // pool_size
    out_w = W // pool_size
    
    # Initialize output
    output = [[0 for _ in range(out_w)] for _ in range(out_h)]
    
    # Perform max pooling
    for i in range(out_h):
        for j in range(out_w):
            max_val = float('-inf')
            
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[i * pool_size + a][j * pool_size + b]
                    if val > max_val:
                        max_val = val
            
            output[i][j] = max_val
    
    return output