import math

def gaussian_naive_bayes(X_train, y_train, X_test):
    n = len(X_train)
    d = len(X_train[0])
    
    # group samples by class
    classes = {}
    for x, y in zip(X_train, y_train):
        classes.setdefault(y, []).append(x)
    
    means = {}
    variances = {}
    priors = {}
    eps = 1e-9
    
    # compute statistics for each class
    for c, samples in classes.items():
        nc = len(samples)
        priors[c] = nc / n
        
        mean = []
        var = []
        
        for j in range(d):
            vals = [s[j] for s in samples]
            m = sum(vals) / nc
            v = sum((v - m) ** 2 for v in vals) / nc
            mean.append(m)
            var.append(v + eps)
        
        means[c] = mean
        variances[c] = var
    
    predictions = []
    
    # classify test points
    for x in X_test:
        best_class = None
        best_log_prob = -float("inf")
        
        for c in classes:
            log_prob = math.log(priors[c])
            
            for j in range(d):
                mu = means[c][j]
                var = variances[c][j]
                
                log_prob += -0.5 * math.log(2 * math.pi * var)
                log_prob += -((x[j] - mu) ** 2) / (2 * var)
            
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c
        
        predictions.append(best_class)
    
    return predictions