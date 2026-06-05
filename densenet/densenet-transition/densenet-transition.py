import torch
import torch.nn.functional as F

def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    def to64(a):
        return torch.tensor(a, dtype=torch.float64)

    x           = to64(x)
    gamma       = to64(bn_gamma).view(1, -1, 1, 1)
    beta        = to64(bn_beta).view(1, -1, 1, 1)
    mean        = to64(bn_mean).view(1, -1, 1, 1)
    var         = to64(bn_var).view(1, -1, 1, 1)
    conv_weight = to64(conv_weight)

    # BN → ReLU → Conv 1×1 → AvgPool 2×2
    x = gamma * (x - mean) / (var + eps).sqrt() + beta
    x = F.relu(x)
    x = F.conv2d(x, conv_weight, bias=None, stride=1, padding=0)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    return x