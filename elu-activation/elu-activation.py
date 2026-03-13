import math

def elu(x, alpha=1.0):
    result = []
    
    for v in x:
        if v > 0:
            result.append(v)
        else:
            result.append(alpha * (math.exp(v) - 1))
    
    return result


# Example 1
x = [1.0, -1.0, 0.0, 2.0, -0.5]
alpha = 1.0
print([round(v,4) for v in elu(x, alpha)])

# Example 2
x = [-1.0, -2.0, -3.0]
alpha = 2.0
print([round(v,4) for v in elu(x, alpha)])