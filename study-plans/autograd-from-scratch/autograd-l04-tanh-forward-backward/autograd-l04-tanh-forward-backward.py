import torch

def tanh_forward_backward(x, upstream_gradient):
    output = torch.tanh(x)
    local_gradient = 1 - output.square()
    input_gradient = upstream_gradient * local_gradient

    return output, input_gradient