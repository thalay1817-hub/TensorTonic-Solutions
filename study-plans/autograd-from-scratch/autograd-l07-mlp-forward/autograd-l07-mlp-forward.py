import torch
from typing import List, Tuple

def mlp_forward(
    inputs: torch.Tensor,
    weights: List[torch.Tensor],
    biases: List[torch.Tensor],
    dtype: torch.dtype = None
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    # Determine promoted floating dtype among inputs and all weights/biases
    promoted_dtype = inputs.dtype
    for w in weights:
        promoted_dtype = torch.promote_types(promoted_dtype, w.dtype)
    for b in biases:
        promoted_dtype = torch.promote_types(promoted_dtype, b.dtype)

    # Cast input without modifying original
    activation = inputs.to(dtype=promoted_dtype)

    layer_outputs = []

    for W, b in zip(weights, biases):
        W_c = W.to(dtype=promoted_dtype)
        b_c = b.to(dtype=promoted_dtype)

        preactivation = torch.matmul(W_c, activation) + b_c
        activation = torch.tanh(preactivation)

        layer_outputs.append(activation)

    return activation, layer_outputs


# Example tests
if __name__ == "__main__":
    result = mlp_forward(
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        [torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)],
        [torch.tensor([0.0, 0.0], dtype=torch.float64)]
    )
    print(result)
    # Expected: ([0.7615941559557649, 0.9640275800758169], [[0.7615941559557649, 0.9640275800758169]])

    result = mlp_forward(
        torch.tensor([1.0], dtype=torch.float64),
        [
            torch.tensor([[1.0], [-1.0]], dtype=torch.float64),
            torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        ],
        [
            torch.tensor([0.0, 0.0], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64)
        ]
    )
    print(result)
    # Expected: ([0.0], [[0.7615941559557649, -0.7615941559557649], [0.0]])

    result = mlp_forward(
        torch.tensor([10.0], dtype=torch.float64),
        [torch.tensor([[2.0]], dtype=torch.float64)],
        [torch.tensor([0.0], dtype=torch.float64)]
    )
    print(result)
    # Expected: ([1.0], [[1.0]])