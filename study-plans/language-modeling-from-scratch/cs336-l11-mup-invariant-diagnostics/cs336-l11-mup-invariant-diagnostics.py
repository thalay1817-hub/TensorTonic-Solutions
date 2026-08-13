import torch

def mup_invariant_diagnostics(
    widths,
    activation_rms,
    activation_change_rms,
    slope_tolerance,
):
    widths64 = widths.to(torch.float64)
    activation64 = activation_rms.to(torch.float64)
    change64 = activation_change_rms.to(torch.float64)

    log_widths = torch.log(widths64)
    ones = torch.ones_like(log_widths)
    design = torch.stack([ones, log_widths], dim=1)

    def fit(y):
        log_y = torch.log(y)
        solution = torch.linalg.lstsq(design, log_y.unsqueeze(1)).solution
        a = float(solution[0, 0].item())
        b = float(solution[1, 0].item())
        return a, b

    activation_intercept, activation_slope = fit(activation64)
    change_intercept, change_slope = fit(change64)

    activation_ratio = float((activation64.max() / activation64.min()).item())
    change_ratio = float((change64.max() / change64.min()).item())

    activation_invariant = abs(activation_slope) <= slope_tolerance
    change_invariant = abs(change_slope) <= slope_tolerance

    return {
        "activation_slope": activation_slope,
        "activation_intercept": activation_intercept,
        "change_slope": change_slope,
        "change_intercept": change_intercept,
        "activation_ratio": activation_ratio,
        "change_ratio": change_ratio,
        "activation_invariant": bool(activation_invariant),
        "change_invariant": bool(change_invariant),
    }