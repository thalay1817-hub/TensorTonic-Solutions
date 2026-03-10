import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    # Convert inputs to numpy arrays
    x_t = np.asarray(x_t, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    Wx = np.asarray(Wx, dtype=float)
    Wh = np.asarray(Wh, dtype=float)
    b = np.asarray(b, dtype=float)

    # RNN hidden state update
    h_t = np.tanh(x_t @ Wx + h_prev @ Wh + b)

    return h_t


# Example 1
x_t = [1.0, 0.0]
h_prev = [0.0, 0.0]
Wx = np.eye(2)        # Identity matrix
Wh = np.zeros((2,2))  # Zero matrix
b = np.zeros(2)

print(rnn_step_forward(x_t, h_prev, Wx, Wh, b))


# Example 2
x_t = [0.0, 0.0]
h_prev = [1.0, -1.0]
Wx = np.zeros((2,2))
Wh = np.eye(2)
b = np.zeros(2)

print(rnn_step_forward(x_t, h_prev, Wx, Wh, b))
