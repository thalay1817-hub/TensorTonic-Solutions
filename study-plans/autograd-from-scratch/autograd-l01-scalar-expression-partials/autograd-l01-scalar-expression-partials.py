import numpy as np

def scalar_expression_partials(a, b, c, h):
    a = np.float64(a)
    b = np.float64(b)
    c = np.float64(c)
    h = np.float64(h)

    def expr(a, b, c):
        return a * b + c

    d = expr(a, b, c)
    partial_a = (expr(a + h, b, c) - d) / h
    partial_b = (expr(a, b + h, c) - d) / h
    partial_c = (expr(a, b, c + h) - d) / h

    return (float(d), float(partial_a), float(partial_b), float(partial_c))