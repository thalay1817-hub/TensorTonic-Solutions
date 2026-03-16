import numpy as np

def matrix_trace(A):
    A = np.asarray(A)
    n = A.shape[0]
    
    trace = 0
    for i in range(n):
        trace += A[i, i]
        
    return trace


# Examples
print(matrix_trace([[1, 2], [3, 4]]))
print(matrix_trace([[2, -1, 0], [3, 5, 1], [0, 2, -2]]))
print(matrix_trace([[42]]))
