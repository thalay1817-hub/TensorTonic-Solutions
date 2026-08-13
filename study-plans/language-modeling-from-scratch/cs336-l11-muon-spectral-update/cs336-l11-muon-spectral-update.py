import torch

def muon_spectral_update(
    parameter,
    gradient,
    previous_momentum,
    momentum_coefficient,
    learning_rate,
):
    orig_dtype = parameter.dtype
    device = parameter.device

    new_momentum = momentum_coefficient * previous_momentum + gradient

    momentum_f32 = new_momentum.to(torch.float32)
    u, _, vh = torch.linalg.svd(momentum_f32, full_matrices=False)
    orthogonalized_f32 = u @ vh

    orthogonalized_update = orthogonalized_f32.to(orig_dtype)

    new_parameter = parameter - learning_rate * orthogonalized_update

    return {
        "new_parameter": new_parameter.to(orig_dtype).to(device),
        "new_momentum": new_momentum.to(orig_dtype).to(device),
        "orthogonalized_update": orthogonalized_update.to(orig_dtype).to(device),
    }