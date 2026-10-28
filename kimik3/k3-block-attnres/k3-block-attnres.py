import torch


def _depth_read(embedding, folded, pseudo_query, eps):
 
    sources = torch.cat([embedding.unsqueeze(0), folded], dim=0)  # (S, B, T, D)

    rms = torch.sqrt(sources.pow(2).mean(dim=-1, keepdim=True) + eps)
    keys = sources / rms

    w = pseudo_query.to(device=sources.device, dtype=sources.dtype)
    scores = torch.einsum("sbtd,d->sbt", keys, w)

    weights = torch.softmax(scores, dim=0)
    retrieved = torch.einsum("sbt,sbtd->btd", weights, sources)
    return retrieved, weights, sources


def block_attention_residual(embedding, previous_outputs, pseudo_query,
                             block_size, eps=1e-6):
    embedding = torch.as_tensor(embedding)
    if not embedding.is_floating_point():
        embedding = embedding.to(torch.get_default_dtype())
    pseudo_query = torch.as_tensor(pseudo_query, device=embedding.device)

    # Empty list / empty tensor -> zero layers
    if (not torch.is_tensor(previous_outputs)
            and len(previous_outputs) == 0) or (
            torch.is_tensor(previous_outputs) and previous_outputs.numel() == 0):
        previous_outputs = embedding.new_zeros((0,) + tuple(embedding.shape))
    else:
        previous_outputs = torch.as_tensor(previous_outputs).to(
            device=embedding.device, dtype=embedding.dtype
        )

    block_size = int(block_size)

    num_layers = previous_outputs.shape[0]
    num_complete = num_layers // block_size
    remainder = num_layers % block_size

    sums = []
    for j in range(num_complete):
        s = j * block_size
        sums.append(previous_outputs[s:s + block_size].sum(dim=0))

    if remainder > 0:
        sums.append(previous_outputs[num_complete * block_size:].sum(dim=0))

    if sums:
        folded = torch.stack(sums, dim=0)
    else:
        folded = embedding.new_zeros((0,) + tuple(embedding.shape))

    retrieved, weights, block_sources = _depth_read(
        embedding, folded, pseudo_query, eps
    )

    return {
        "retrieved_representation": retrieved,
        "attention_weights": weights,
        "block_sources": block_sources,
    }