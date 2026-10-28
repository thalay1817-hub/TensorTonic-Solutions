import torch


def full_attention_residual(embedding, previous_outputs, pseudo_query, eps=1e-6):
    
    sources = torch.cat(
        [embedding.unsqueeze(0), previous_outputs.to(embedding.dtype)], dim=0
    )  

    rms = torch.sqrt(sources.pow(2).mean(dim=-1, keepdim=True) + eps)
    keys = sources / rms

    w = pseudo_query.to(device=sources.device, dtype=sources.dtype)
    scores = torch.einsum("sbtd,d->sbt", keys, w)

    attention_weights = torch.softmax(scores, dim=0)

 
    retrieved = torch.einsum("sbt,sbtd->btd", attention_weights, sources)

    return {
        "retrieved_representation": retrieved,
        "attention_weights": attention_weights,
    }