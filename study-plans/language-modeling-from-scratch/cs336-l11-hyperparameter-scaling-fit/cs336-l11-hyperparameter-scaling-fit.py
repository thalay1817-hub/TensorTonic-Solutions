import torch

def hyperparameter_scaling_fit(model_scales, learning_rates, batch_sizes, measured_losses, target_scale):
    scales = model_scales.detach().clone()
    lrs = learning_rates.detach().clone()
    bss = batch_sizes.detach().clone()
    losses = measured_losses.detach().clone()

    num_scales = scales.shape[0]
    num_lr = lrs.shape[0]
    num_bs = bss.shape[0]

    optima = []
    opt_lr_list = []
    opt_bs_list = []
    log_scale_list = []

    for i in range(num_scales):
        grid = losses[i].reshape(-1)
        idx = int(torch.argmin(grid).item())
        lr_idx = idx // num_bs
        bs_idx = idx % num_bs

        lr_val = lrs[lr_idx].item()
        bs_val = bss[bs_idx].item()
        loss_val = losses[i, lr_idx, bs_idx].item()
        scale_val = scales[i].item()

        optima.append({
            "model_scale": float(scale_val),
            "learning_rate": float(lr_val),
            "batch_size": float(bs_val),
            "loss": float(loss_val),
        })

        opt_lr_list.append(lr_val)
        opt_bs_list.append(bs_val)
        log_scale_list.append(scale_val)

    log_scale = torch.log(torch.tensor(log_scale_list, dtype=torch.float64))
    log_lr = torch.log(torch.tensor(opt_lr_list, dtype=torch.float64))
    log_bs = torch.log(torch.tensor(opt_bs_list, dtype=torch.float64))

    X = torch.stack([torch.ones_like(log_scale), log_scale], dim=1)

    sol_lr = torch.linalg.lstsq(X, log_lr.unsqueeze(1)).solution
    log_a_lr = sol_lr[0, 0].item()
    b_lr = sol_lr[1, 0].item()

    sol_bs = torch.linalg.lstsq(X, log_bs.unsqueeze(1)).solution
    log_a_bs = sol_bs[0, 0].item()
    b_bs = sol_bs[1, 0].item()

    a_lr = float(torch.exp(torch.tensor(log_a_lr, dtype=torch.float64)).item())
    a_bs = float(torch.exp(torch.tensor(log_a_bs, dtype=torch.float64)).item())

    pred_log_lr = log_a_lr + b_lr * log_scale
    pred_log_bs = log_a_bs + b_bs * log_scale

    learning_rate_residuals = (log_lr - pred_log_lr).to(torch.float64)
    batch_size_residuals = (log_bs - pred_log_bs).to(torch.float64)

    target_log_scale = torch.log(torch.tensor(float(target_scale), dtype=torch.float64)).item()
    target_learning_rate = float(a_lr * (float(target_scale) ** b_lr))
    target_batch_size = float(a_bs * (float(target_scale) ** b_bs))

    return {
        "optima": optima,
        "learning_rate_coefficient": a_lr,
        "learning_rate_exponent": float(b_lr),
        "batch_size_coefficient": a_bs,
        "batch_size_exponent": float(b_bs),
        "learning_rate_residuals": learning_rate_residuals,
        "batch_size_residuals": batch_size_residuals,
        "target_learning_rate": target_learning_rate,
        "target_batch_size": target_batch_size,
    }