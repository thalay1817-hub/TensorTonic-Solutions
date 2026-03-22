def polynomial_features(values, degree):
    result = []
    
    for x in values:
        row = [x ** p for p in range(degree + 1)]
        result.append(row)
    
    return result