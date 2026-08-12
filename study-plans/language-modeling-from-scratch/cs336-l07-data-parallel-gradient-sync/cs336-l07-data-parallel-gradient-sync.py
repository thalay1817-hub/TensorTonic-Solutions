import torch
from typing import List, Optional, Dict


def data_parallel_gradient_sync(
    parameter_replicas: List[List[torch.Tensor]],
    gradient_replicas: List[List[Optional[torch.Tensor]]],
    learning_rate: float,
) -> Dict[str, List]:
   
    if not isinstance(parameter_replicas, list) or len(parameter_replicas) == 0:
        raise ValueError("parameter_replicas must be a nonempty list")
    if not isinstance(gradient_replicas, list) or len(gradient_replicas) == 0:
        raise ValueError("gradient_replicas must be a nonempty list")
    if len(parameter_replicas) != len(gradient_replicas):
        raise ValueError("parameter_replicas and gradient_replicas must have the same rank count")

    R = len(parameter_replicas)

    num_params = len(parameter_replicas[0])
    if num_params == 0:
        raise ValueError("Every rank must contain at least one parameter tensor")

    for r in range(R):
        if len(parameter_replicas[r]) != num_params:
            raise ValueError("All ranks must have the same number of parameter tensors")
        if len(gradient_replicas[r]) != num_params:
            raise ValueError("gradient_replicas must match parameter_replicas structure")

    supported_dtypes = (torch.float32, torch.bfloat16)

    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
        raise ValueError("learning_rate must be a Python number")
    learning_rate = float(learning_rate)
    if learning_rate < 0.0 or learning_rate != learning_rate or learning_rate in (
        float("inf"),
        float("-inf"),
    ):
        raise ValueError("learning_rate must be finite and nonnegative")

    mean_gradients: List[torch.Tensor] = []
    updated_reference: List[torch.Tensor] = []

    for j in range(num_params):
        base_param = parameter_replicas[0][j]

        if base_param.dtype not in supported_dtypes:
            raise ValueError(f"Unsupported parameter dtype: {base_param.dtype}")

        shape = base_param.shape
        dtype = base_param.dtype
        device = base_param.device

        # Validate consistency across ranks for this parameter index.
        for r in range(R):
            p = parameter_replicas[r][j]
            if p.shape != shape or p.dtype != dtype or p.device != device:
                raise ValueError(
                    "Corresponding parameter replicas must match in shape, dtype, and device"
                )
            g = gradient_replicas[r][j]
            if g is not None:
                if g.shape != shape or g.dtype != dtype or g.device != device:
                    raise ValueError(
                        "Each gradient must be None or match its parameter tensor"
                    )

        # Accumulate in float32 regardless of param dtype (float32 or bfloat16).
        acc = torch.zeros(shape, dtype=torch.float32, device=device)
        for r in range(R):
            g = gradient_replicas[r][j]
            if g is not None:
                acc = acc + g.to(torch.float32)

        mean_grad_f32 = acc / float(R)

        base_param_f32 = base_param.to(torch.float32)
        updated_f32 = base_param_f32 - learning_rate * mean_grad_f32

        mean_grad_out = mean_grad_f32.to(dtype)
        updated_out = updated_f32.to(dtype)

        mean_gradients.append(mean_grad_out)
        updated_reference.append(updated_out)

    # Every rank receives an independent copy of the same update.
    updated_replicas: List[List[torch.Tensor]] = [
        [t.clone() for t in updated_reference] for _ in range(R)
    ]

    return {
        "mean_gradients": mean_gradients,
        "updated_replicas": updated_replicas,
    }