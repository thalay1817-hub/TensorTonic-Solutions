import numpy as np

def linear_regression_closed_form(X, y):
    # Convert inputs to NumPy arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Apply normal equation
    w = np.linalg.inv(X.T @ X) @ X.T @ y

    return w


# Example 1
X = [[1], [2], [3]]
y = [2, 4, 6]
print(linear_regression_closed_form(X, y))

# Example 2
X = [[1,1], [1,2], [1,3]]
y = [3,5,7]
print(linear_regression_closed_form(X, y))