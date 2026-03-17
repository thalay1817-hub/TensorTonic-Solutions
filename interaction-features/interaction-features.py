def interaction_features(X):
    result = []
    
    for row in X:
        # Copy original features
        new_row = list(row)
        d = len(row)
        
        # Generate pairwise interactions (i < j)
        for i in range(d):
            for j in range(i + 1, d):
                new_row.append(row[i] * row[j])
        
        result.append(new_row)
    
    return result


# Example usage
X1 = [[1, 2, 3]]
X2 = [[1, 2], [3, 4]]

print(interaction_features(X1))
print(interaction_features(X2))