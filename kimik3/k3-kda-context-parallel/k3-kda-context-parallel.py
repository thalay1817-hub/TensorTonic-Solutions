import torch


def kda_context_parallel(transition_matrices, local_contributions, initial_state):
    M = torch.as_tensor(transition_matrices)
    if not M.is_floating_point():
        M = M.to(torch.get_default_dtype())
    dev, dt = M.device, M.dtype

    U = torch.as_tensor(local_contributions).to(device=dev, dtype=dt)
    S0 = torch.as_tensor(initial_state).to(device=dev, dtype=dt)

    n = M.shape[0]

    incoming = []
    outgoing = []

    # exclusive prefix composition, carried as one affine map (pm, pu)
    width = M.shape[-1]
    pm = torch.eye(width, device=dev, dtype=dt)
    pu = torch.zeros_like(S0)

    for i in range(n):
        # state produced by all preceding segments
        s_in = torch.matmul(pm, S0) + pu
        s_out = torch.matmul(M[i], s_in) + U[i]
        incoming.append(s_in)
        outgoing.append(s_out)

        # compose current map on the left: (M_i, U_i) o (pm, pu)
        pu = torch.matmul(M[i], pu) + U[i]
        pm = torch.matmul(M[i], pm)

    incoming_states = torch.stack(incoming, dim=0)
    outgoing_states = torch.stack(outgoing, dim=0)
    final_state = outgoing_states[-1]

    return {
        "incoming_states": incoming_states,
        "outgoing_states": outgoing_states,
        "final_state": final_state,
    }