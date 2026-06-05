import torch
import torch.nn.functional as F

def densenet_forward(x, weights, growth_rate, eps):
    x = torch.tensor(x, dtype=torch.float32)
    
    def t(v):
        return torch.tensor(v, dtype=torch.float32)
    
    # Stem conv (no BN, no bias)
    h = F.conv2d(x, t(weights["stem_conv"]), padding=1)
    
    blocks = weights["blocks"]
    transitions = weights["transitions"]
    
    for i, block in enumerate(blocks):
        # Dense block
        feats = [h]
        for layer in block:
            concat = torch.cat(feats, dim=1)
            out = F.batch_norm(concat, t(layer["bn_mean"]), t(layer["bn_var"]),
                               t(layer["bn_gamma"]), t(layer["bn_beta"]),
                               training=False, eps=eps)
            out = F.relu(out)
            out = F.conv2d(out, t(layer["conv_weight"]), padding=1)
            feats.append(out)
        h = torch.cat(feats, dim=1)
        
        # Transition (not after last block)
        if i < len(blocks) - 1:
            tr = transitions[i]
            h = F.batch_norm(h, t(tr["bn_mean"]), t(tr["bn_var"]),
                             t(tr["bn_gamma"]), t(tr["bn_beta"]),
                             training=False, eps=eps)
            h = F.relu(h)
            h = F.conv2d(h, t(tr["conv_weight"]))   # 1x1, no padding
            h = F.avg_pool2d(h, 2, stride=2)
    
    # Final BN-ReLU
    h = F.batch_norm(h,
                     t(weights["final_bn_mean"]), t(weights["final_bn_var"]),
                     t(weights["final_bn_gamma"]), t(weights["final_bn_beta"]),
                     training=False, eps=eps)
    h = F.relu(h)
    
    # Global average pool
    p = h.mean(dim=(2, 3))
    
    # Linear classifier
    logits = p @ t(weights["fc_weight"]).T + t(weights["fc_bias"])
    
    return logits