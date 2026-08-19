import torch
from typing import List, Tuple

def train_tiny_micrograd_mlp(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: List[torch.Tensor],
    biases: List[torch.Tensor],
    learning_rate: float,
    steps: int
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    # Determine promoted floating dtype across all tensors
    promoted_dtype = inputs.dtype
    promoted_dtype = torch.promote_types(promoted_dtype, targets.dtype)
    for W in weights:
        promoted_dtype = torch.promote_types(promoted_dtype, W.dtype)
    for b in biases:
        promoted_dtype = torch.promote_types(promoted_dtype, b.dtype)

    device = inputs.device

    X = inputs.to(dtype=promoted_dtype)
    y = targets.to(dtype=promoted_dtype)
    y_col = y.reshape(-1, 1)  # (B, 1)

    # Working copies of parameters (do not mutate originals)
    W_list = [W.to(dtype=promoted_dtype).clone() for W in weights]
    b_list = [b.to(dtype=promoted_dtype).clone() for b in biases]

    lr = torch.tensor(learning_rate, dtype=promoted_dtype, device=device)

    num_layers = len(W_list)

    def forward(X_in, Ws, bs):
        activations = [X_in]  # H_0 ... H_L
        H = X_in
        for W, b in zip(Ws, bs):
            preact = torch.matmul(H, W.t()) + b
            H = torch.tanh(preact)
            activations.append(H)
        return activations

    loss_history = []

    for _ in range(steps):
        # ---- Forward pass (fresh) ----
        activations = forward(X, W_list, b_list)
        H_final = activations[-1]  # (B, 1)

        # Pre-update loss
        error = H_final - y_col
        pre_loss = torch.sum(error * error)
        loss_history.append(pre_loss)

        # ---- Backward pass: fresh gradient tensors ----
        G = 2.0 * error  # (B, 1), gradient wrt final activation

        dW_list = [None] * num_layers
        db_list = [None] * num_layers

        for l in range(num_layers - 1, -1, -1):
            H_l = activations[l + 1]        # current layer's activation
            H_prev = activations[l]         # preceding activation (input to this layer)
            W_l = W_list[l]

            D = G * (1.0 - H_l * H_l)       # (B, out_l)

            dW = torch.matmul(D.t(), H_prev)   # (out_l, in_l)
            db = torch.sum(D, dim=0)           # (out_l,)

            dW_list[l] = dW
            db_list[l] = db

            if l > 0:
                G = torch.matmul(D, W_l)       # (B, in_l) -> becomes G for layer l-1

        # ---- Simultaneous parameter update ----
        new_W_list = [W_list[l] - lr * dW_list[l] for l in range(num_layers)]
        new_b_list = [b_list[l] - lr * db_list[l] for l in range(num_layers)]

        W_list = new_W_list
        b_list = new_b_list

    # ---- Final forward pass after last update ----
    final_activations = forward(X, W_list, b_list)
    final_H = final_activations[-1]  # (B, 1)
    final_predictions = final_H.reshape(-1)  # (B,)

    final_error = final_H - y_col
    final_loss = torch.sum(final_error * final_error)

    return final_predictions, final_loss, W_list, b_list, loss_history


# Example tests
if __name__ == "__main__":
    result = train_tiny_micrograd_mlp(
        torch.tensor([[1.0], [-1.0]], dtype=torch.float64),
        torch.tensor([1.0, -1.0], dtype=torch.float64),
        [torch.tensor([[0.5]], dtype=torch.float64)],
        [torch.tensor([0.0], dtype=torch.float64)],
        0.1,
        0
    )
    print(result[0].tolist(), result[1].item(),
          [W.tolist() for W in result[2]], [b.tolist() for b in result[3]],
          [l.item() for l in result[4]])
    # Expected: ([0.46211715726000974,-0.46211715726000974], 0.578635905028106, [[[0.5]]], [[0.0]], [])

    result = train_tiny_micrograd_mlp(
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        [torch.tensor([[0.0]], dtype=torch.float64)],
        [torch.tensor([0.0], dtype=torch.float64)],
        0.1,
        1
    )
    print(result[0].tolist(), result[1].item(),
          [W.tolist() for W in result[2]], [b.tolist() for b in result[3]],
          [l.item() for l in result[4]])
    # Expected: ([0.3799489622552249], 0.38446328940837254, [[[0.2]]], [[0.2]], [1.0])

    result = train_tiny_micrograd_mlp(
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        [torch.tensor([[1.0]], dtype=torch.float64)],
        [torch.tensor([0.0], dtype=torch.float64)],
        0.05,
        2
    )
    print(result[0].tolist(), result[1].item(),
          [W.tolist() for W in result[2]], [b.tolist() for b in result[3]],
          [l.item() for l in result[4]])
    # Expected: ([0.7004812181336727], 0.49067393695803396, [[[0.9341223582874592]]], [[-0.06587764171254085]], [0.5800256583859739, 0.5378651649748326])