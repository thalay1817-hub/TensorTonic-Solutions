import torch

def neuron_gradient_check(inputs, weights, bias, h):
    device = weights.device if weights.numel() > 0 else bias.device

    inputs = inputs.to(torch.float64)
    weights = weights.to(torch.float64)
    bias = bias.to(torch.float64)
    h = torch.tensor(h, dtype=torch.float64, device=device)

    def forward(w, b):
        preact = torch.sum(inputs * w) + b
        return torch.tanh(preact)

    # Analytic gradients
    y = forward(weights, bias)
    local = 1 - y.square()
    analytic_weight_gradients = local * inputs
    analytic_bias_gradient = local

    # Numerical weight gradients: perturb one weight at a time
    n = weights.shape[0]
    numerical_weight_gradients = torch.empty(n, dtype=torch.float64, device=device)
    for i in range(n):
        w_perturbed = weights.clone()
        w_perturbed[i] = w_perturbed[i] + h
        y_perturbed = forward(w_perturbed, bias)
        numerical_weight_gradients[i] = (y_perturbed - y) / h

    # Numerical bias gradient
    b_perturbed = bias + h
    y_b_perturbed = forward(weights, b_perturbed)
    numerical_bias_gradient = (y_b_perturbed - y) / h

    # Max absolute disagreement
    diffs = []
    if n > 0:
        diffs.append(torch.max(torch.abs(analytic_weight_gradients - numerical_weight_gradients)))
    diffs.append(torch.abs(analytic_bias_gradient - numerical_bias_gradient))
    max_error = torch.stack(diffs).max()

    return (
        analytic_weight_gradients,
        numerical_weight_gradients,
        analytic_bias_gradient,
        numerical_bias_gradient,
        max_error,
    )