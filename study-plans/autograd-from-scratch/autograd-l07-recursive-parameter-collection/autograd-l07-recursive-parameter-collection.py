import torch
from typing import List, Tuple

def recursive_parameter_collection(
    weights: List[torch.Tensor],
    biases: List[torch.Tensor]
) -> Tuple[List[torch.Tensor], int]:
    params = []

    for W, b in zip(weights, biases):
        num_neurons, num_inputs = W.shape
        for j in range(num_neurons):
            for i in range(num_inputs):
               
                params.append(W[j, i])
            
            params.append(b[j])

    return params, len(params)



if __name__ == "__main__":
    result = recursive_parameter_collection(
        [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)],
        [torch.tensor([5.0, 6.0], dtype=torch.float64)]
    )
    print(([p.item() for p in result[0]], result[1]))
   
    result = recursive_parameter_collection(
        [
            torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.float64),
            torch.tensor([[3.0, 4.0]], dtype=torch.float64)
        ],
        [
            torch.tensor([0.5, -0.5], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64)
        ]
    )
    print(([p.item() for p in result[0]], result[1]))
   
    result = recursive_parameter_collection(
        [torch.tensor([[-2.5]], dtype=torch.float64)],
        [torch.tensor([0.25], dtype=torch.float64)]
    )
    print(([p.item() for p in result[0]], result[1]))
   