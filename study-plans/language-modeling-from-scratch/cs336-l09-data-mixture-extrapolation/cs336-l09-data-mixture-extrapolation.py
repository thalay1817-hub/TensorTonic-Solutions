import torch

def data_mixture_extrapolation(mixture_names, data_sizes, losses, target_size, floor_candidates):
    def fit_single_curve(D, L, floors):
        log_D = torch.log(D)
        min_loss = torch.min(L).item()

        best = None  # (rmse, E, A, alpha)

        for i in range(floors.shape[0]):
            E = floors[i].item()

            if not (E < min_loss):
                continue

            diff = L - E
            if torch.any(diff <= 0):
                continue

            log_diff = torch.log(diff)

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

            pred_obs = E + A * torch.pow(D, -alpha)
            rmse = torch.sqrt(torch.mean((pred_obs - L) ** 2)).item()

            if best is None or rmse < best[0]:
                best = (rmse, E, A, alpha)

        if best is None:
            raise ValueError("No valid floor candidate produced a positive-decay fit for this mixture.")

        rmse, E, A, alpha = best
        return E, A, alpha, rmse

    D_all = data_sizes.detach().to(torch.float64).clone()
    L_all = losses.detach().to(torch.float64).clone()
    floors = floor_candidates.detach().to(torch.float64).clone()
    target = float(target_size)

    num_mixtures = D_all.shape[0]

    fits = []
    predictions_list = []

    for k in range(num_mixtures):
        D = D_all[k].contiguous()
        L = L_all[k].contiguous()

        E, A, alpha, rmse = fit_single_curve(D, L, floors)

        pred_target = E + A * (target ** (-alpha))

        fits.append({
            "name": mixture_names[k],
            "floor": float(E),
            "coefficient": float(A),
            "exponent": float(alpha),
            "rmse": float(rmse),
        })
        predictions_list.append(pred_target)

    target_predictions = torch.tensor(predictions_list, dtype=torch.float64)

    winning_index = min(
        range(len(predictions_list)),
        key=lambda index: (predictions_list[index], index)
    )
    winning_mixture = mixture_names[winning_index]

    return {
        "fits": fits,
        "target_predictions": target_predictions,
        "winning_mixture": winning_mixture,
        "winning_index": winning_index,
    }