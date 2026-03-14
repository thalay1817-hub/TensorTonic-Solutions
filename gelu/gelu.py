import numpy as np
import math

def gelu(x):
    x = np.asarray(x, dtype=float)

    # make erf work element-wise
    erf_vec = np.vectorize(math.erf)

    return 0.5 * x * (1 + erf_vec(x / np.sqrt(2)))


# Examples
print(np.round(gelu([-1.0, 0.0, 1.0]), 6))

print(np.round(gelu([[-2., -1.], [0., 1.]]), 6))