import torch
import torch.nn.functional as F

def bottleneck_layer(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight,
                     bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight, eps=1e-5):
    def to64(a):
        return torch.tensor(a, dtype=torch.float64)

    def bn(x, gamma, mean, var, beta):
        g = gamma.view(1, -1, 1, 1)
        m = mean.view(1, -1, 1, 1)
        v = var.view(1, -1, 1, 1)
        b = beta.view(1, -1, 1, 1)
        return g * (x - m) / (v + eps).sqrt() + b

    x            = to64(x)
    bn1_gamma    = to64(bn1_gamma);  bn1_beta = to64(bn1_beta)
    bn1_mean     = to64(bn1_mean);   bn1_var  = to64(bn1_var)
    conv1_weight = to64(conv1_weight)
    bn2_gamma    = to64(bn2_gamma);  bn2_beta = to64(bn2_beta)
    bn2_mean     = to64(bn2_mean);   bn2_var  = to64(bn2_var)
    conv2_weight = to64(conv2_weight)

    # Stage 1: BN → ReLU → Conv 1×1  (C → 4k)
    y1 = F.relu(bn(x, bn1_gamma, bn1_mean, bn1_var, bn1_beta))
    y1 = F.conv2d(y1, conv1_weight, bias=None, stride=1, padding=0)

    # Stage 2: BN → ReLU → Conv 3×3  (4k → k)
    y2 = F.relu(bn(y1, bn2_gamma, bn2_mean, bn2_var, bn2_beta))
    y2 = F.conv2d(y2, conv2_weight, bias=None, stride=1, padding=1)

    return y2