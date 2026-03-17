import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test, alpha=1.0):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    
    n_train, d = X_train.shape
    classes = np.unique(y_train)
    n_classes = len(classes)
    
    # Compute class priors log P(y)
    log_priors = {}
    theta = {}  # P(x_i=1 | y)
    
    for c in classes:
        X_c = X_train[y_train == c]
        n_c = len(X_c)
        
        # Prior
        log_priors[c] = np.log(n_c / n_train)
        
        # Likelihood with Laplace smoothing
        # θ = (count(x_i=1) + α) / (n_c + 2α)
        theta[c] = (np.sum(X_c, axis=0) + alpha) / (n_c + 2 * alpha)
    
    # Compute log posterior for each test sample
    results = []
    
    for x in X_test:
        row = []
        for c in classes:
            log_prob = log_priors[c]
            
            # log P(x|y)
            log_prob += np.sum(
                x * np.log(theta[c]) +
                (1 - x) * np.log(1 - theta[c])
            )
            
            row.append(round(log_prob, 4))
        
        results.append(row)
    
    return results


# Example usage
print(naive_bayes_bernoulli(
    [[1, 0], [0, 1]],
    [1, 0],
    [[1, 0]]
))

print(naive_bayes_bernoulli(
    [[1, 0], [1, 1], [0, 0], [0, 1]],
    [0, 0, 1, 1],
    [[1, 0], [0, 1]]
))