import torch


def mxfp4_expert_linear(latent_tokens, packed_weights, scale_bytes,
                  selected_experts, mixture_weights, shared_output):
    x = torch.as_tensor(latent_tokens)
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    dev, dt = x.device, x.dtype

    packed = torch.as_tensor(packed_weights).to(dev).to(torch.int64)
    scales = torch.as_tensor(scale_bytes).to(dev).to(torch.int64)
    sel = torch.as_tensor(selected_experts).to(dev).to(torch.int64)
    p = torch.as_tensor(mixture_weights).to(device=dev, dtype=dt)
    shared = torch.as_tensor(shared_output).to(device=dev, dtype=dt)

    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=dev, dtype=dt,
    )

    T, K = sel.shape
    E, O, G, _ = packed.shape
    I = 32 * G
    
    unique = torch.unique(sel)
    out = shared.clone()

    for e in unique.tolist():
        pk = packed[e]                                   
        low = lut[pk & 0xF]                              
        high = lut[(pk >> 4) & 0xF]                
        vals = torch.stack([low, high], dim=-1)         
        vals = vals.reshape(O, G, 32)                    

        scale = torch.pow(
            torch.tensor(2.0, device=dev, dtype=dt),
            (scales[e] - 127).to(dt),
        )                                                
        W = (vals * scale.unsqueeze(-1)).reshape(O, I)   

        y = torch.matmul(x, W.transpose(0, 1))           
        weight = (p * (sel == e).to(dt)).sum(dim=1)     
        out = out + weight.unsqueeze(-1) * y

    return out