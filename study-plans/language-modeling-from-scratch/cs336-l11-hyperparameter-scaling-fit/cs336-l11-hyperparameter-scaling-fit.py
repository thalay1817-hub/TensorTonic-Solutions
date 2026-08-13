import torch

def hyperparameter_scaling_fit(
    model_scales,
    learning_rates,
    batch_sizes,
    measured_losses,
    target_scale,
):
    num_scales = measured_losses.shape[0]
    num_lr = measured_losses.shape[1]
    num_bs = measured_losses.shape[2]

    optima = []
    selected_lrs = []
    selected_bss = []
    scales_list = []

    for i in range(num_scales):
        grid = measured_losses[i]
        flat = grid.reshape(-1)
        idx = int(torch.argmin(flat).item())
        lr_idx = idx // num_bs
        bs_idx = idx % num_bs

        scale_val = model_scales[i].item() if torch.is_tensor(model_scales) else model_scales[i]
        lr_val = float(learning_rates[lr_idx].item())
        bs_val = float(batch_sizes[bs_idx].item())
        loss_val = float(grid[lr_idx, bs_idx].item())

        optima.append({
            "model_scale": scale_val,
            "learning_rate": lr_val,
            "batch_size": bs_val,
            "loss": loss_val,
        })

        scales_list.append(float(scale_val))
        selected_lrs.append(lr_val)
        selected_bss.append(bs_val)

    scales_t = torch.tensor(scales_list, dtype=torch.float64)
    lrs_t = torch.tensor(selected_lrs, dtype=torch.float64)
    bss_t = torch.tensor(selected_bss, dtype=torch.float64)

    log_scales = torch.log(scales_t)
    ones = torch.ones_like(log_scales)
    design = torch.stack([ones, log_scales], dim=1)

    def fit(log_y):
        solution = torch.linalg.lstsq(design, log_y.unsqueeze(1)).solution
        log_a = solution[0, 0]
        b = solution[1, 0]
        residuals = log_y - (log_a + b * log_scales)
        a = torch.exp(log_a)
        return float(a.item()), float(b.item()), residuals

    log_lrs = torch.log(lrs_t)
    log_bss = torch.log(bss_t)

    lr_a, lr_b, lr_residuals = fit(log_lrs)
    bs_a, bs_b, bs_residuals = fit(log_bss)

    target_scale_val = float(target_scale)
    target_lr = lr_a * (target_scale_val ** lr_b)
    target_bs = bs_a * (target_scale_val ** bs_b)

    return {
        "optima": optima,
        "learning_rate_coefficient": lr_a,
        "learning_rate_exponent": lr_b,
        "batch_size_coefficient": bs_a,
        "batch_size_exponent": bs_b,
        "learning_rate_residuals": lr_residuals,
        "batch_size_residuals": bs_residuals,
        "target_learning_rate": target_lr,
        "target_batch_size": target_bs,
    }