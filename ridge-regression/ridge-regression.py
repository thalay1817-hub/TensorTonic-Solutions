import numpy as np

def ridge_regression(X, y, lam):
    X = np.array(X)
    y = np.array(y)

    # X transpose
    Xt = X.T

    # Identity matrix of size d x d
    d = X.shape[1]
    I = np.eye(d)

    # Ridge formula
    w = np.linalg.inv(Xt @ X + lam * I) @ Xt @ y

    return w


# Example 1
X = [[1, 0], [0, 1]]
y = [3, 5]
lam = 1.0

print(ridge_regression(X, y, lam))
# Output: [1.5 2.5]


# Example 2
X = [[1, 1], [1, 2], [1, 3]]
y = [3, 5, 7]
lam = 0.0

print(ridge_regression(X, y, lam))
# Output: [1. 2.]