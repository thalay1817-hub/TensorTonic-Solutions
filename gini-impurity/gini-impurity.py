import numpy as np

def gini_impurity(y_left, y_right):
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    # Helper function to compute Gini for one node
    def gini(y):
        n = len(y)
        if n == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return 1.0 - np.sum(probs ** 2)
    
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    
    if n_total == 0:
        return 0.0
    
    # Weighted Gini
    gini_total = (n_left / n_total) * gini(y_left) + \
                 (n_right / n_total) * gini(y_right)
    
    return float(gini_total)