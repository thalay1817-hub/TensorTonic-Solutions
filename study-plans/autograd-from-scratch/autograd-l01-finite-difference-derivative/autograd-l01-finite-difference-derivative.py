import numpy as np

def finite_difference_derivative(coefficients, x, h):
    coeffs = np.array(coefficients, dtype=np.float64)
    x = np.float64(x)
    h = np.float64(h)

    def horner(coeffs, point):
        result = np.float64(0.0)
        for c in coeffs[::-1]:
            result = result * point + c
        return result

    fx = horner(coeffs, x)
    fxh = horner(coeffs, x + h)
    slope = (fxh - fx) / h

    return (float(fx), float(fxh), float(slope))