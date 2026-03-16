def catalog_coverage(recommendations, n_items):
    unique_items = set()
    
    # collect all recommended items
    for rec_list in recommendations:
        for item in rec_list:
            unique_items.add(item)
    
    # compute coverage
    coverage = len(unique_items) / n_items
    
    return coverage


# Example 1
recommendations = [[1,2,3],[2,3,4],[4,5,6]]
n_items = 10
print(catalog_coverage(recommendations, n_items))

# Example 2
recommendations = [[1,2],[1,2],[1,2]]
n_items = 5
print(catalog_coverage(recommendations, n_items))