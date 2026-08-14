import torch

def fit_data_power_law(data_sizes, losses, target_sizes, floor_candidates):
    D = data_sizes.detach().to(torch.float64).clone()
    L = losses.detach().to(torch.float64).clone()
    targets = target_sizes.detach().to(torch.float64).clone()
    floors = floor_candidates.detach().to(torch.float64).clone()

    log_D = torch.log(D)
    min_loss = torch.min(L).item()

    best = None  # (rmse, floor, A, alpha)

    for i in range(floors.shape[0]):
        E = floors[i].item()

        # Floor must be strictly below every observed loss
        if not (E < min_loss):
            continue

        diff = L - E
        if torch.any(diff <= 0):
            continue

        log_diff = torch.log(diff)

        # Design matrix: [1, log(D)]
        X = torch.stack([torch.ones_like(log_D), log_D], dim=1)

        try:
            sol = torch.linalg.lstsq(X, log_diff.unsqueeze(1)).solution
        except RuntimeError:
            continue

        a = sol[0, 0].item()
        b = sol[1, 0].item()

        A = float(torch.exp(torch.tensor(a, dtype=torch.float64)).item())
        alpha = -b

        if not (torch.isfinite(torch.tensor(A)) and torch.isfinite(torch.tensor(alpha))):
            continue
        if not (A > 0 and alpha > 0):
            continue

        # Predict on observed data to compute RMSE on raw loss
        pred_obs = E + A * torch.pow(D, -alpha)
        rmse = torch.sqrt(torch.mean((pred_obs - L) ** 2)).item()

        if best is None or rmse < best[0]:
            best = (rmse, E, A, alpha)

    if best is None:
        raise ValueError("No valid floor candidate produced a positive-decay fit.")

    rmse, E, A, alpha = best

    predictions = E + A * torch.pow(targets, -alpha)
    predictions = predictions.to(torch.float64)

    return {
        "floor": float(E),
        "coefficient": float(A),
        "exponent": float(alpha),
        "predictions": predictions,
        "rmse": float(rmse),
    }