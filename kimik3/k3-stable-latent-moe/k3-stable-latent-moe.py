import torch


def situ_glu(x, w_gate, w_up, w_down, gate_cap, up_cap):
    g = torch.matmul(x, w_gate.transpose(-1, -2))
    u = torch.matmul(x, w_up.transpose(-1, -2))
    gate_branch = gate_cap * torch.tanh(g / gate_cap) * torch.sigmoid(g)
    up_branch = up_cap * torch.tanh(u / up_cap)
    h = gate_branch * up_branch
    return torch.matmul(h, w_down.transpose(-1, -2))


def stable_latent_moe(tokens, latent_down_projection, latent_up_projection,
                      router_projection, current_bias,
                      routed_gate_weights, routed_up_weights,
                      routed_down_weights,
                      shared_gate_weights, shared_up_weights,
                      shared_down_weights,
                      selected_count, eps=1e-6, gate_cap=4.0, up_cap=25.0):
    x = torch.as_tensor(tokens)
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    dev, dt = x.device, x.dtype

    def cast(t):
        return torch.as_tensor(t).to(device=dev, dtype=dt)

    w_down = cast(latent_down_projection)
    w_up = cast(latent_up_projection)
    w_router = cast(router_projection)
    bias = cast(current_bias)
    rg, ru, rd = (cast(routed_gate_weights), cast(routed_up_weights),
                  cast(routed_down_weights))
    sg, su, sd = (cast(shared_gate_weights), cast(shared_up_weights),
                  cast(shared_down_weights))
    k = int(selected_count)

    # routing
    r = torch.sigmoid(torch.matmul(x, w_router.transpose(-1, -2)))
    _, selected_experts = torch.topk(r + bias, k, dim=-1)
    selected_raw = torch.gather(r, 1, selected_experts)
    mixture_weights = selected_raw / selected_raw.sum(dim=-1, keepdim=True)

    # latent routed experts
    z = torch.matmul(x, w_down.transpose(-1, -2))
    g = torch.einsum("tl,ehl->teh", z, rg)
    u = torch.einsum("tl,ehl->teh", z, ru)
    h = (gate_cap * torch.tanh(g / gate_cap) * torch.sigmoid(g)
         * up_cap * torch.tanh(u / up_cap))
    expert_out = torch.einsum("teh,eoh->teo", h, rd)

    idx = selected_experts.unsqueeze(-1).expand(-1, -1, expert_out.shape[-1])
    chosen = torch.gather(expert_out, 1, idx)
    latent_routed_aggregate = (mixture_weights.unsqueeze(-1) * chosen).sum(1)

    # RMSNorm and up-projection
    rms = torch.sqrt(latent_routed_aggregate.pow(2).mean(-1, keepdim=True) + eps)
    normed = latent_routed_aggregate / rms
    routed_out = torch.matmul(normed, w_up.transpose(-1, -2))

    # two shared experts at model width
    shared_out = situ_glu(x, sg[0], su[0], sd[0], gate_cap, up_cap) \
        + situ_glu(x, sg[1], su[1], sd[1], gate_cap, up_cap)

    output = shared_out + routed_out

    return {
        "output": output,
        "selected_experts": selected_experts,
        "mixture_weights": mixture_weights,
        "latent_routed_aggregate": latent_routed_aggregate,
    }