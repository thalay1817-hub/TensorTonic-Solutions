import torch

def critical_batch_size(batch_sizes, steps_to_target, reached_mask):
    examples = [
        float(batch_sizes[i].item() * steps_to_target[i].item()) if reached_mask[i].item() else None
        for i in range(len(batch_sizes))
    ]

    reached_examples = [e for e in examples if e is not None]
    reached_steps = [
        float(steps_to_target[i].item())
        for i in range(len(steps_to_target))
        if reached_mask[i].item()
    ]

    min_steps = float(min(reached_steps))
    min_examples = float(min(reached_examples))
    critical_batch_size = float(min_examples / min_steps)

    return {
        "examples": examples,
        "min_steps": min_steps,
        "min_examples": min_examples,
        "critical_batch_size": critical_batch_size,
    }