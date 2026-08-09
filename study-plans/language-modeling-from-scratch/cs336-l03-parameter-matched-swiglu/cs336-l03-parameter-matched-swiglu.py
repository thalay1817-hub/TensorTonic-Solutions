import math
import torch
import torch.nn.functional as F


def parameter_matched_swiglu(x, w_g, w_v, w_o, base_params):
    orig_dtype = x.dtype
    d = x.shape[-1]
    h_max = w_g.shape[-1]

    # Determine budget-matched hidden width
    h = math.floor(base_params / (3 * d) + 0.5)
    h = max(1, min(h, h_max))

    parameter_count = 3 * d * h

    # Slice weights to selected hidden width (does not mutate originals)
    w_g_sliced = w_g[:, :h]
    w_v_sliced = w_v[:, :h]
    w_o_sliced = w_o[:h, :]

    # Compute in float32
    x_f32 = x.to(torch.float32)
    w_g_f32 = w_g_sliced.to(torch.float32)
    w_v_f32 = w_v_sliced.to(torch.float32)
    w_o_f32 = w_o_sliced.to(torch.float32)

    gate = x_f32 @ w_g_f32
    value = x_f32 @ w_v_f32
    activated = F.silu(gate) * value
    result = activated @ w_o_f32

    output = result.to(orig_dtype)

    return {
        "output": output,
        "hidden_width": h,
        "parameter_count": parameter_count,
    }