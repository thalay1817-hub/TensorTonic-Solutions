import torch

def dense_layer_forward(inputs: torch.Tensor, weight_matrix: torch.Tensor, biases: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    # Determine promoted floating dtype among inputs, weight_matrix, and biases
    promoted_dtype = torch.promote_types(inputs.dtype, weight_matrix.dtype)
    promoted_dtype = torch.promote_types(promoted_dtype, biases.dtype)

    # Cast to promoted dtype without modifying originals
    inputs_c = inputs.to(dtype=promoted_dtype)
    weight_matrix_c = weight_matrix.to(dtype=promoted_dtype)
    biases_c = biases.to(dtype=promoted_dtype)

    # Matrix-vector multiplication: each row dot the input vector
    preactivations = torch.matmul(weight_matrix_c, inputs_c) + biases_c

    if nonlinear:
        output = torch.tanh(preactivations)
    else:
        output = preactivations

    return output


# Example tests
if __name__ == "__main__":
    print(dense_layer_forward(
        torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64),
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]], dtype=torch.float64),
        torch.tensor([0.5, -0.5], dtype=torch.float64),
        False
    ))  # Expected: [0.5, 3.5]

    print(dense_layer_forward(
        torch.tensor([2.0, 1.0], dtype=torch.float64),
        torch.tensor([[1.0, -1.0]], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        True
    ))  # Expected: [0.7615941559557649]

    print(dense_layer_forward(
        torch.tensor([3.0, -2.0], dtype=torch.float64),
        torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        False
    ))  # Expected: [0.0, 1.0]