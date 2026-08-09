import torch


def rotary_embed(q, k, positions, inv_freq):
    def rotate(x, theta_cos, theta_sin):
        x_f32 = x.to(torch.float32)
        x_even = x_f32[..., 0::2]
        x_odd = x_f32[..., 1::2]

        rotated_even = x_even * theta_cos - x_odd * theta_sin
        rotated_odd = x_even * theta_sin + x_odd * theta_cos

        # Interleave even/odd back together
        stacked = torch.stack((rotated_even, rotated_odd), dim=-1)
        out = stacked.flatten(start_dim=-2)
        return out.to(x.dtype)

    positions_f32 = positions.to(torch.float32)
    inv_freq_f32 = inv_freq.to(torch.float32)

    # Reshape positions to broadcast against (B, H, S, D/2)
    if positions_f32.dim() == 1:
        # shape (S) -> (1, 1, S, 1)
        pos_reshaped = positions_f32.view(1, 1, -1, 1)
    else:
        # shape (B, S) -> (B, 1, S, 1)
        B, S = positions_f32.shape
        pos_reshaped = positions_f32.view(B, 1, S, 1)

    # inv_freq shape (D/2) -> broadcast automatically as last dim
    theta = pos_reshaped * inv_freq_f32.view(1, 1, 1, -1)

    theta_cos = torch.cos(theta)
    theta_sin = torch.sin(theta)

    q_rotated = rotate(q, theta_cos, theta_sin)
    k_rotated = rotate(k, theta_cos, theta_sin)

    return {
        "q_rotated": q_rotated,
        "k_rotated": k_rotated,
    }