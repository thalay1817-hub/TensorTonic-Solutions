import torch

def neuron_forward(inputs, weights, bias):
    preactivation = torch.sum(inputs * weights) + bias
    output = torch.tanh(preactivation)

    return preactivation, output