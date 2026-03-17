import math

def gaussian_kernel(size, sigma):
    # Initialize kernel
    kernel = [[0.0 for _ in range(size)] for _ in range(size)]
    
    center = size // 2
    total_sum = 0.0

    # Step 1: Compute unnormalized Gaussian values
    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center
            value = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel[i][j] = value
            total_sum += value

    # Step 2: Normalize so sum = 1
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total_sum

    return kernel


# Example usage
size = 3
sigma = 1.0

result = gaussian_kernel(size, sigma)

# Print rounded output
for row in result:
    print([round(val, 4) for val in row])