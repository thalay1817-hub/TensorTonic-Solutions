import numpy as np

def gradient_check_product_chain(a, b, c, f, h):
    a = np.float64(a)
    b = np.float64(b)
    c = np.float64(c)
    f = np.float64(f)
    h = np.float64(h)

    def loss(a, b, c, f):
        e = a * b + c
        return e * f

    L = loss(a, b, c, f)

    # Analytic gradients
    # e = a*b + c, L = e*f
    # dL/da = f*b, dL/db = f*a, dL/dc = f, dL/df = e
    e = a * b + c
    grad_a = f * b
    grad_b = f * a
    grad_c = f
    grad_f = e

    analytic = [grad_a, grad_b, grad_c, grad_f]

    # Numerical gradients via forward difference
    num_a = (loss(a + h, b, c, f) - L) / h
    num_b = (loss(a, b + h, c, f) - L) / h
    num_c = (loss(a, b, c + h, f) - L) / h
    num_f = (loss(a, b, c, f + h) - L) / h

    numerical = [num_a, num_b, num_c, num_f]

    max_diff = max(abs(an - nu) for an, nu in zip(analytic, numerical))

    return (
        float(L),
        [float(g) for g in analytic],
        [float(g) for g in numerical],
        float(max_diff),
    )