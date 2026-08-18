import torch
from functools import reduce

def neuron_backward(inputs, weights, bias, upstream_gradient):
    # Determine the promoted floating dtype across all inputs
    common_dtype = reduce(
        torch.promote_types,
        [inputs.dtype, weights.dtype, bias.dtype, upstream_gradient.dtype]
    )

    inputs = inputs.to(common_dtype)
    weights = weights.to(common_dtype)
    bias = bias.to(common_dtype)
    upstream_gradient = upstream_gradient.to(common_dtype)

    preactivation = torch.sum(inputs * weights) + bias
    output = torch.tanh(preactivation)

    delta = upstream_gradient * (1 - output.square())

    input_gradients = delta * weights
    weight_gradients = delta * inputs
    bias_gradient = delta

    return output, input_gradients, weight_gradients, bias_gradient