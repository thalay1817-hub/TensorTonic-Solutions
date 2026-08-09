import torch


def gradient_accumulation_step(param, microbatch_inputs, microbatch_targets, lr):
    # Work on a fresh leaf tensor so the original param/grad are untouched
    param_work = param.detach().clone().to(dtype=param.dtype, device=param.device)
    param_work.requires_grad_(True)

    # Total example count across all microbatches
    total_n = sum(len(targets) for targets in microbatch_targets)

    for X_m, y_m in zip(microbatch_inputs, microbatch_targets):
        X = torch.as_tensor(X_m, dtype=param.dtype, device=param.device)
        y = torch.as_tensor(y_m, dtype=param.dtype, device=param.device)

        preds = X @ param_work
        # Scale so accumulated gradient equals gradient of full mean-squared error
        loss = ((preds - y) ** 2).sum() / total_n
        loss.backward()

    full_grad = param_work.grad.detach().clone()

    with torch.no_grad():
        new_param = param_work - lr * full_grad
    new_param = new_param.detach().clone()

    return {
        "new_param": new_param,
        "full_grad": full_grad,
    }