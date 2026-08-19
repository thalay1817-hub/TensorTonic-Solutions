import torch
from typing import Tuple

def squared_error_loss_gradients(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
   
    promoted_dtype = torch.promote_types(predictions.dtype, targets.dtype)

   
    predictions_c = predictions.to(dtype=promoted_dtype)
    targets_c = targets.to(dtype=promoted_dtype)

   
    error = predictions_c - targets_c

   
    loss = torch.sum(error * error)
    loss = loss.reshape(())

   
    gradient = 2.0 * error

    return loss, gradient



if __name__ == "__main__":
    result = squared_error_loss_gradients(
        torch.tensor([1.0, -1.0], dtype=torch.float64),
        torch.tensor([1.0, 1.0], dtype=torch.float64)
    )
    print((result[0].item(), result[1].tolist()))
    

    result = squared_error_loss_gradients(
        torch.tensor([0.5], dtype=torch.float64),
        torch.tensor([0.5], dtype=torch.float64)
    )
    print((result[0].item(), result[1].tolist()))
    

    result = squared_error_loss_gradients(
        torch.tensor([], dtype=torch.float64),
        torch.tensor([], dtype=torch.float64)
    )
    print((result[0].item(), result[1].tolist()))
   