import math

def cyclic_encoding(values, period):
    result = []
    
    for v in values:
        angle = 2 * math.pi * v / period
        result.append([math.sin(angle), math.cos(angle)])
    
    return result


# Example usage
values1 = [0, 6, 12, 18]
period1 = 24
print(cyclic_encoding(values1, period1))

values2 = [0, 1, 2, 3]
period2 = 4
print(cyclic_encoding(values2, period2))