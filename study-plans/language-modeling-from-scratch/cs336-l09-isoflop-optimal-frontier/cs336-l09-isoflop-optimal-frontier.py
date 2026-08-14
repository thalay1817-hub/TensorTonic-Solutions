import torch

def isoflop_optimal_frontier(compute_budgets, parameter_counts, terminal_losses):
    C = compute_budgets.detach().to(torch.float64).clone()
    N_all = parameter_counts.detach().to(torch.float64).clone()
    L_all = terminal_losses.detach().to(torch.float64).clone()

    num_budgets = C.shape[0]

    optima = []
    log_N_opt_list = []
    log_C_list = []

    for i in range(num_budgets):
        Ci = C[i].item()
        N = N_all[i].contiguous()
        L = L_all[i].contiguous()

        x = torch.log(N)

        X = torch.stack([x * x, x, torch.ones_like(x)], dim=1)

        sol = torch.linalg.lstsq(X, L.unsqueeze(1)).solution
        a = sol[0, 0].item()
        b = sol[1, 0].item()
        c = sol[2, 0].item()

        x_star = -b / (2.0 * a)
        N_star = float(torch.exp(torch.tensor(x_star, dtype=torch.float64)).item())
        D_star = Ci / (6.0 * N_star)
        loss_star = a * x_star * x_star + b * x_star + c

        optima.append({
            "compute": float(Ci),
            "parameters": float(N_star),
            "tokens": float(D_star),
            "loss": float(loss_star),
        })

        log_N_opt_list.append(x_star)
        log_C_list.append(float(torch.log(torch.tensor(Ci, dtype=torch.float64)).item()))

    log_N_opt = torch.tensor(log_N_opt_list, dtype=torch.float64)
    log_C = torch.tensor(log_C_list, dtype=torch.float64)

    Xf = torch.stack([torch.ones_like(log_C), log_C], dim=1)
    sol_f = torch.linalg.lstsq(Xf, log_N_opt.unsqueeze(1)).solution
    log_k = sol_f[0, 0].item()
    alpha = sol_f[1, 0].item()

    frontier_scale = float(torch.exp(torch.tensor(log_k, dtype=torch.float64)).item())
    frontier_exponent = float(alpha)

    return {
        "optima": optima,
        "frontier_scale": frontier_scale,
        "frontier_exponent": frontier_exponent,
    }