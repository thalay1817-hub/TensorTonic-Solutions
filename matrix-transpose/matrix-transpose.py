import numpy as np

def matrix_transpose(A):
    A = np.asarray(A)
    
    n, m = A.shape   # rows, columns
    
    # Create new matrix with shape (m, n)
    T = np.zeros((m, n), dtype=A.dtype)
    
    # Fill using nested loop
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    
    return T


# Examples
A1 = [[1, 2, 3], [4, 5, 6]]
print(matrix_transpose(A1))

A2 = [[1, 2], [3, 4]]
print(matrix_transpose(A2))

A3 = [[1, 2, 3, 4]]
print(matrix_transpose(A3))