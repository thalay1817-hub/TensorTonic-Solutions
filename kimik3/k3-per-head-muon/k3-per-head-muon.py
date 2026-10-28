import torch


def per_head_muon(parameter, gradient, previous_momentum, num_heads,
                  momentum_coefficient, learning_rate):
    theta = torch.as_tensor(parameter)
    if not theta.is_floating_point():
        theta = theta.to(torch.get_default_dtype())
    dev, dt = theta.device, theta.dtype

    grad = torch.as_tensor(gradient).to(device=dev, dtype=dt)
    prev = torch.as_tensor(previous_momentum).to(device=dev, dtype=dt)

    mu = float(momentum_coefficient)
    lr = float(learning_rate)
    h = int(num_heads)

    momentum = mu * prev + grad

    rows = momentum.shape[0]
    block = rows // h

    factors = []
    for i in range(h):
        m_h = momentum[i * block:(i + 1) * block]
        u, s, vh = torch.linalg.svd(m_h, full_matrices=False)
        factors.append(torch.matmul(u, vh))

    orthogonalized = torch.cat(factors, dim=0)

    updated_parameter = theta - lr * orthogonalized

    return {
        "updated_parameter": updated_parameter,
        "updated_momentum": momentum,
        "orthogonalized_update": orthogonalized,
    }