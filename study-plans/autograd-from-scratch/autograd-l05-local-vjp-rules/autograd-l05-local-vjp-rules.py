import numpy as np

def local_vjp(operation, inputs, output, upstream_gradient):
    inputs = [np.float64(v) for v in inputs]
    output = np.float64(output)
    g = np.float64(upstream_gradient)

    if operation == "add":
        a, b = inputs
        return [float(g), float(g)]
    elif operation == "mul":
        a, b = inputs
        return [float(g * b), float(g * a)]
    elif operation == "tanh":
        t = output
        return [float(g * (1 - t * t))]
    else:
        raise ValueError(f"Unsupported operation: {operation}")