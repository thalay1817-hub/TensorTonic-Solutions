import numpy as np

def t_test_one_sample(x, mu0):
    # Convert to NumPy array
    x = np.asarray(x, dtype=float)

    n = len(x)

    # Sample mean
    mean = np.mean(x)

    # Sample standard deviation (Bessel correction)
    s = np.sqrt(np.sum((x - mean) ** 2) / (n - 1))

    # Standard error
    se = s / np.sqrt(n)

    # t-statistic
    t_stat = (mean - mu0) / se

    return t_stat


# Example 1
x = [2.1, 2.4, 1.9, 2.6, 2.0]
print(round(t_test_one_sample(x, 2.0), 2))

# Example 2
x = [3.0, 5.0]
print(t_test_one_sample(x, 4.0))

# Example 3
x = [1.0, 1.5, 2.0]
print(round(t_test_one_sample(x, 3.0), 2))