import torch

def muon_spectral_update(parameter, gradient, previous_momentum, momentum_coefficient, learning_rate):
    orig_dtype = parameter.dtype
    device = parameter.device

    W = parameter.detach().clone()
    G = gradient.detach().clone()
    B_prev = previous_momentum.detach().clone()

    # Compute new momentum
    new_momentum = momentum_coefficient * B_prev + G

    # Compute SVD in float32
    B_f32 = new_momentum.to(torch.float32)
    u, _, vh = torch.linalg.svd(B_f32, full_matrices=False)

    O = u @ vh  # orthogonalized update in float32

    orthogonalized_update = O.to(orig_dtype).to(device)

    new_parameter = (W - learning_rate * O.to(orig_dtype)).to(orig_dtype).to(device)
    new_momentum = new_momentum.to(orig_dtype).to(device)

    return {
        "new_parameter": new_parameter,
        "new_momentum": new_momentum,
        "orthogonalized_update": orthogonalized_update,
    }
