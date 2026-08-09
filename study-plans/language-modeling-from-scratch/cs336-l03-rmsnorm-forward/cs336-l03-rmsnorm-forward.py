import torch


def rmsnorm(x, g, epsilon):
    orig_dtype = x.dtype

    # Compute mean square and normalization in float32 for numerical stability
    x_f32 = x.to(torch.float32)
    g_f32 = g.to(torch.float32)

    mean_square = x_f32.pow(2).mean(dim=-1, keepdim=True)
    normalized = x_f32 / torch.sqrt(mean_square + epsilon)

    scaled = normalized * g_f32

    # Cast once back to the original input dtype
    return scaled.to(orig_dtype)