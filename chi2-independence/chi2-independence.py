import numpy as np

def chi2_independence (C):
    # Convert to numpy array
    C = np.asarray(C, dtype=float)

    # Row sums, column sums, total
    row_totals = np.sum(C, axis=1)
    col_totals = np.sum(C, axis=0)
    total = np.sum(C)

    # Expected frequency table
    expected = np.outer(row_totals, col_totals) / total

    # Chi-square statistic
    chi2 = np.sum((C - expected) ** 2 / expected)

    return chi2, expected


# Example 1
C = [[10,20],[20,10]]
chi2, expected = chi2_independence(C)
print("chi2 =", round(chi2,3))
print("expected =", expected)

# Example 2
C = [[20,30],[40,60]]
print(chi2_independence(C))

# Example 3
C = [[25,25],[25,25]]
print(chi2_independence(C))