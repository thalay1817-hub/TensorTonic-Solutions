import torch

def critical_batch_size(batch_sizes, steps_to_target, reached_mask):
    B = batch_sizes.detach().clone()
    S = steps_to_target.detach().clone()
    R = reached_mask.detach().clone()

    n = B.shape[0]

    examples = []
    reached_examples = []
    reached_steps = []

    for i in range(n):
        if bool(R[i].item()):
            b = B[i].item()
            s = S[i].item()
            e = float(b * s)
            examples.append(e)
            reached_examples.append(e)
            reached_steps.append(float(s))
        else:
            examples.append(None)

    min_steps = float(min(reached_steps))
    min_examples = float(min(reached_examples))
    critical_batch_size = float(min_examples / min_steps)

    return {
        "examples": examples,
        "min_steps": min_steps,
        "min_examples": min_examples,
        "critical_batch_size": critical_batch_size,
    }