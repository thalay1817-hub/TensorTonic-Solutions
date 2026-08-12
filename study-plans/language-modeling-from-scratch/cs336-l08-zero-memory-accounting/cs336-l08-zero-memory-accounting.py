def zero_memory_accounting(
    parameter_bytes: int,
    gradient_bytes: int,
    optimizer_bytes: int,
    world_size: int,
) -> dict:
    
    for name, value in (
        ("parameter_bytes", parameter_bytes),
        ("gradient_bytes", gradient_bytes),
        ("optimizer_bytes", optimizer_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be a Python integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")

    if not isinstance(world_size, int) or isinstance(world_size, bool):
        raise ValueError("world_size must be a Python integer")
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    P = parameter_bytes
    G = gradient_bytes
    O = optimizer_bytes
    R = world_size

    def ceil_div(value: int, divisor: int) -> int:
        return (value + divisor - 1) // divisor

    ddp = P + G + O
    zero1 = P + G + ceil_div(O, R)
    zero2 = P + ceil_div(G + O, R)
    zero3 = ceil_div(P + G + O, R)

    return {
        "ddp": int(ddp),
        "zero1": int(zero1),
        "zero2": int(zero2),
        "zero3": int(zero3),
    }