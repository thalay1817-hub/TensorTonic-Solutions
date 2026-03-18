def ordinal_encoding(values, ordering):
    # Create mapping: category -> index
    mapping = {v: i for i, v in enumerate(ordering)}
    
    # Encode values
    return [mapping[v] for v in values]


# Example usage
print(ordinal_encoding(["low", "medium", "high", "medium"], ["low", "medium", "high"]))
print(ordinal_encoding(["S", "M", "L", "XL", "S"], ["S", "M", "L", "XL"]))