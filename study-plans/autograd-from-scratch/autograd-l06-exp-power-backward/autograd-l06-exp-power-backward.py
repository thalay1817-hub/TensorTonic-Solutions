import numpy as np

def exp_power_backward(x, exponent, upstream_gradient):
    x = np.float64(x)
    exponent = np.float64(exponent)
    upstream_gradient = np.float64(upstream_gradient)

    with np.errstate(all='ignore'):
        # Exponential
        exp_output = np.exp(x)
        exp_gradient = upstream_gradient * exp_output

        # Power
        if exponent == 0.0:
            power_output = np.float64(1.0)
            power_gradient = np.float64(0.0)
        else:
            power_output = np.power(x, exponent)
            power_gradient = upstream_gradient * exponent * np.power(x, exponent - 1.0)

    return (float(exp_output), float(exp_gradient), float(power_output), float(power_gradient))


# Example tests
if __name__ == "__main__":
    print(exp_power_backward(2.0, 0.0, 3.0))
    print(exp_power_backward(2.0, -2.0, -0.5))
    print(exp_power_backward(-2.0, 3.0, 2.0))