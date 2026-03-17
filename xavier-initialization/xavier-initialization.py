import math

def xavier_initialization(W, fan_in, fan_out):
    # Compute limit
    limit = math.sqrt(6 / (fan_in + fan_out))
    
    # Scale weights
    result = []
    for row in W:
        new_row = []
        for val in row:
            scaled = val * 2 * limit - limit
            new_row.append(round(scaled, 4))
        result.append(new_row)
    
    return result


# Example usage
print(xavier_initialization([[0.5, 0.5], [0.5, 0.5]], 2, 2))
print(xavier_initialization([[0, 1], [1, 0]], 2, 2))