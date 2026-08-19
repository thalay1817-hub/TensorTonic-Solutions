import torch

def scalar_neuron_module(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    # Determine promoted floating dtype among inputs, weights, and bias
    promoted_dtype = torch.promote_types(inputs.dtype, weights.dtype)
    promoted_dtype = torch.promote_types(promoted_dtype, bias.dtype)

    # Cast to promoted dtype without modifying originals
    inputs_c = inputs.to(dtype=promoted_dtype)
    weights_c = weights.to(dtype=promoted_dtype)
    bias_c = bias.to(dtype=promoted_dtype)

    # Weighted sum (handles empty tensors correctly -> 0.0)
    weighted_sum = torch.sum(inputs_c * weights_c)

    # Preactivation
    a = weighted_sum + bias_c

    # Ensure scalar shape
    a = a.reshape(())

    if nonlinear:
        o = torch.tanh(a)
    else:
        o = a

    return o


# Example tests
if __name__ == "__main__":
    print(scalar_neuron_module(
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        torch.tensor([0.5, -1.0], dtype=torch.float64),
        torch.tensor(0.25, dtype=torch.float64),
        False
    ))  # Expected: -1.25

    print(scalar_neuron_module(
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        True
    ))  # Expected: 0.0

    print(scalar_neuron_module(
        torch.tensor([], dtype=torch.float64),
        torch.tensor([], dtype=torch.float64),
        torch.tensor(1.5, dtype=torch.float64),
        True
    ))  # Expected: 0.9051482536448664