def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    x = float(x0)

    for _ in range(steps):
        grad = 2 * a * x + b   # derivative of ax^2 + bx + c
        x = x - lr * grad      # update rule

    return x


# Example 1
print(round(gradient_descent_quadratic(1, -4, 3, 0, 0.1, 50), 4))

# Example 2
print(round(gradient_descent_quadratic(0.5, -1, 0, -5, 0.2, 100), 4))