import math


def memory_accountant(param_shapes, param_bytes_per_element,
                            grad_bytes_per_element,
                            activation_shapes, activation_bytes_per_element,
                            optimizer, optimizer_bytes_per_element):
    def total_elements(shapes):
        count = 0
        for shape in shapes:
            elements = 1
            for dim in shape:
                elements *= dim
            count += elements
        return count

    # Total element counts (does not mutate input shape lists)
    param_elements = total_elements(param_shapes)
    activation_elements = total_elements(activation_shapes)

    parameters = param_elements * param_bytes_per_element
    gradients = param_elements * grad_bytes_per_element
    activations = activation_elements * activation_bytes_per_element

    optimizer_multiplier = {"sgd": 0, "adagrad": 1, "adam": 2}[optimizer]
    optimizer_state = param_elements * optimizer_multiplier * optimizer_bytes_per_element

    total = parameters + gradients + activations + optimizer_state

    return {
        "parameters": parameters,
        "gradients": gradients,
        "activations": activations,
        "optimizer_state": optimizer_state,
        "total": total,
    }