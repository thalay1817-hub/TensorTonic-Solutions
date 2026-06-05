import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    x           = torch.tensor(x,           dtype=torch.float64)
    bn_gamma    = torch.tensor(bn_gamma,    dtype=torch.float64)
    bn_beta     = torch.tensor(bn_beta,     dtype=torch.float64)
    bn_mean     = torch.tensor(bn_mean,     dtype=torch.float64)
    bn_var      = torch.tensor(bn_var,      dtype=torch.float64)
    conv_weight = torch.tensor(conv_weight, dtype=torch.float64)

    # Reshape (C,) vectors to (1, C, 1, 1) for broadcasting
    mean  = bn_mean.view(1, -1, 1, 1)
    var   = bn_var.view(1, -1, 1, 1)
    gamma = bn_gamma.view(1, -1, 1, 1)
    beta  = bn_beta.view(1, -1, 1, 1)

    # BN → ReLU → Conv
    x = gamma * (x - mean) / (var + eps).sqrt() + beta
    x = F.relu(x)
    x = F.conv2d(x, conv_weight, bias=None, stride=1, padding=1)
    return x