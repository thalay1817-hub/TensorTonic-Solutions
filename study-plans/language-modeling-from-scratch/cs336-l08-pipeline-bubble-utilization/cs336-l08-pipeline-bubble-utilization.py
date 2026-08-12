from typing import List, Optional, Dict, Union


def pipeline_bubble_utilization(
    num_stages: int,
    num_microbatches: int,
    include_backward: bool,
) -> Dict[str, Union[List[List[Optional[str]]], int, float]]:
    
    if not isinstance(num_stages, int) or isinstance(num_stages, bool):
        raise ValueError("num_stages must be a Python integer")
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")

    if not isinstance(num_microbatches, int) or isinstance(num_microbatches, bool):
        raise ValueError("num_microbatches must be a Python integer")
    if num_microbatches <= 0:
        raise ValueError("num_microbatches must be positive")

    if not isinstance(include_backward, bool):
        raise ValueError("include_backward must be a Python Boolean")

    p = num_stages
    m = num_microbatches
    T = m + p - 1

    schedule: List[List[Optional[str]]] = []

    # Forward wave, in time-slot order.
    for t in range(T):
        row: List[Optional[str]] = []
        for s in range(p):
            i_forward = t - s
            if 0 <= i_forward < m:
                row.append(f"F{i_forward}")
            else:
                row.append(None)
        schedule.append(row)

    # Backward wave, mirrored, appended when requested.
    if include_backward:
        for u in range(T):
            row = []
            for s in range(p):
                i_backward = m - 1 - (u - (p - 1 - s))
                if 0 <= i_backward < m:
                    row.append(f"B{i_backward}")
                else:
                    row.append(None)
            schedule.append(row)

    q = 2 if include_backward else 1
    useful_slots = q * p * m
    total_slots = q * T * p

    bubble_slots = sum(1 for row in schedule for cell in row if cell is None)

    utilization = float(useful_slots) / float(total_slots) if total_slots > 0 else 0.0

    return {
        "schedule": schedule,
        "bubble_slots": int(bubble_slots),
        "total_slots": int(total_slots),
        "utilization": float(utilization),
    }