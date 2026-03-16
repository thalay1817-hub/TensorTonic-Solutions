def linear_layer_forward(X, W, b):
    n = len(X)          # number of samples
    d_out = len(W[0])   # output dimension
    d_in = len(W)       # input dimension
    
    # Initialize output matrix
    Y = [[0]*d_out for _ in range(n)]
    
    for i in range(n):           # iterate over samples
        for j in range(d_out):   # iterate over output neurons
            s = 0
            for k in range(d_in):   # iterate over input features
                s += X[i][k] * W[k][j]
            Y[i][j] = s + b[j]      # add bias
    
    return Y


# Example 1
print(linear_layer_forward([[1,2],[3,4]], [[1,0],[0,1]], [0,0]))

# Example 2
print(linear_layer_forward([[1,2]], [[1],[2]], [3]))