import torch

def assistant_only_sft_mask(input_ids, role_ids, attention_mask, assistant_role):
    batch, seq_len = input_ids.shape
    device = input_ids.device
    dtype = input_ids.dtype

    labels = torch.full((batch, seq_len), -100, dtype=dtype, device=device)
    loss_mask = torch.zeros((batch, seq_len), dtype=torch.bool, device=device)

    if seq_len <= 1:
        return {"labels": labels, "loss_mask": loss_mask}

    # Source positions: 0..seq_len-2, target positions: 1..seq_len-1
    source_attended = attention_mask[:, :-1]
    target_attended = attention_mask[:, 1:]
    target_is_assistant = (role_ids[:, 1:] == assistant_role)

    valid = source_attended & target_attended & target_is_assistant

    target_tokens = input_ids[:, 1:]

    labels[:, :-1] = torch.where(
        valid,
        target_tokens,
        torch.full_like(target_tokens, -100),
    )
    loss_mask[:, :-1] = valid

    return {"labels": labels, "loss_mask": loss_mask}