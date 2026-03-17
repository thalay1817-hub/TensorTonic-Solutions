def auc(fpr, tpr):
    # Validation
    if len(fpr) != len(tpr):
        raise ValueError("fpr and tpr must have the same length")
    if len(fpr) < 2:
        raise ValueError("At least two points are required")
    
    auc = 0.0
    
    # Trapezoidal rule
    for i in range(len(fpr) - 1):
        width = fpr[i + 1] - fpr[i]
        height = (tpr[i] + tpr[i + 1]) / 2
        auc += width * height
    
    return float(auc)


# Example usage
print(auc([0, 0, 1], [0, 1, 1]))  # 1.0
print(auc([0, 1], [0, 1]))        # 0.5