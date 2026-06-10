import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t, h_prev, C_prev, W_f, W_i, W_c, W_o, b_f, b_i, b_c, b_o):
    combined = np.concatenate([h_prev, x_t], axis=-1)
    f_t     = sigmoid(combined @ W_f.T + b_f)
    i_t     = sigmoid(combined @ W_i.T + b_i)
    c_tilde = np.tanh(combined @ W_c.T + b_c)
    o_t     = sigmoid(combined @ W_o.T + b_o)
    C_t     = f_t * C_prev + i_t * c_tilde
    h_t     = o_t * np.tanh(C_t)
    return h_t, C_t