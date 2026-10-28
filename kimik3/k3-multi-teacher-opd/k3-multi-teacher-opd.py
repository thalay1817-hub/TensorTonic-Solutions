import torch


def multi_teacher_opd_reward(student_logits, teacher_logits, domain_indices,
                      effort_indices, sampled_tokens, clip_threshold):
    student = torch.as_tensor(student_logits)
    if not student.is_floating_point():
        student = student.to(torch.get_default_dtype())
    dev, dt = student.device, student.dtype

    teacher = torch.as_tensor(teacher_logits).to(device=dev, dtype=dt)
    dom = torch.as_tensor(domain_indices).to(device=dev, dtype=torch.int64)
    eff = torch.as_tensor(effort_indices).to(device=dev, dtype=torch.int64)
    tokens = torch.as_tensor(sampled_tokens).to(device=dev, dtype=torch.int64)
    r_max = float(clip_threshold)

    B = student.shape[0]
    batch_idx = torch.arange(B, device=dev)

    # one teacher per batch item: (B, S, V)
    selected = teacher[dom, eff, batch_idx]

    idx = tokens.unsqueeze(-1)
    student_lp = torch.gather(
        torch.log_softmax(student, dim=-1), -1, idx).squeeze(-1)
    teacher_lp = torch.gather(
        torch.log_softmax(selected, dim=-1), -1, idx).squeeze(-1)

    # reward is detached from both policies, then clipped
    gap = (teacher_lp - student_lp).detach()
    rewards = torch.clamp(gap, min=-r_max, max=r_max)

    return {
        "rewards": rewards,
        "teacher_token_log_probs": teacher_lp,
    }