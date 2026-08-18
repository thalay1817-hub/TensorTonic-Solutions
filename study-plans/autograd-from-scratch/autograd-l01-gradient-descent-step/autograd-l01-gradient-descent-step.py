import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    theta = np.array(values, dtype=np.float64)
    grad = np.array(gradients, dtype=np.float64)
    lr = np.float64(learning_rate)

    theta_new = theta - lr * grad
    delta = theta_new - theta
    predicted_change = np.sum(grad * delta)

    return ([float(v) for v in theta_new], float(predicted_change))