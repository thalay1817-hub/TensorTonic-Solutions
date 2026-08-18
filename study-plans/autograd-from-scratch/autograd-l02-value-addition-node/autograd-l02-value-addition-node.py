import numpy as np

def value_addition_node(left, right, output_id):
    x = np.float64(left['data'])
    y = np.float64(right['data'])
    z = x + y

    return {
        'id': output_id,
        'data': float(z),
        'grad': 0.0,
        'op': '+',
        'parents': [left, right]
    }