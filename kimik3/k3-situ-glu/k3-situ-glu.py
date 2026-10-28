import torch


def situ_glu(input_tensor, gate_projection, up_projection, gate_cap, up_cap):
    x = input_tensor
    w_gate = gate_projection.to(device=x.device, dtype=x.dtype)
    w_up = up_projection.to(device=x.device, dtype=x.dtype)

    g = torch.matmul(x, w_gate.transpose(-1, -2))
    u = torch.matmul(x, w_up.transpose(-1, -2))
    
    gate_branch = gate_cap * torch.tanh(g / gate_cap) * torch.sigmoid(g)
    up_branch = up_cap * torch.tanh(u / up_cap)

    return gate_branch * up_branch