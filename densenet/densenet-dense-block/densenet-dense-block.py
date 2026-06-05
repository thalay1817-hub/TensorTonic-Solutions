import torch
import torch.nn.functional as F

def dense_block(x, layers, growth_rate, eps=1e-5):
    def to64(a):
        return torch.tensor(a, dtype=torch.float64)

    def bn_relu_conv(z, layer):
        gamma  = to64(layer['bn_gamma']).view(1, -1, 1, 1)
        beta   = to64(layer['bn_beta']).view(1, -1, 1, 1)
        mean   = to64(layer['bn_mean']).view(1, -1, 1, 1)
        var    = to64(layer['bn_var']).view(1, -1, 1, 1)
        weight = to64(layer['conv_weight'])

        z = gamma * (z - mean) / (var + eps).sqrt() + beta
        z = F.relu(z)
        z = F.conv2d(z, weight, bias=None, stride=1, padding=1)
        return z

    feats = [to64(x)]

    for layer in layers:
        z = torch.cat(feats, dim=1)   # concatenate all previous feature maps
        feats.append(bn_relu_conv(z, layer))

    return torch.cat(feats, dim=1)
