def matrix_factorization_sgd_step(U, V, r, lr, reg):
    
    # Dot product (prediction)
    dot = sum(u * v for u, v in zip(U, V))
    
    # Prediction error
    error = r - dot
    
    # Update using original values
    U_new = [u + lr * (error * v - reg * u) for u, v in zip(U, V)]
    V_new = [v + lr * (error * u - reg * v) for u, v in zip(U, V)]
    
    return U_new, V_new


# Example 1
U = [1, 0]
V = [0, 1]
print(matrix_factorization_sgd_step(U, V, 5.0, 0.1, 0.0))

# Example 2
U = [0.5, 0.3]
V = [0.4, 0.6]
print(matrix_factorization_sgd_step(U, V, 4.0, 0.01, 0.02))