import numpy as np

def poisson_pmf_cdf(lam, k):
    # Helper: log factorial
    def log_factorial(n):
        if n == 0 or n == 1:
            return 0.0
        return np.sum(np.log(np.arange(1, n + 1)))
    
    # PMF using log for stability
    log_pmf = -lam + k * np.log(lam) - log_factorial(k)
    pmf = np.exp(log_pmf)
    
    # CDF = sum of PMFs from 0 to k
    cdf = 0.0
    for i in range(k + 1):
        log_p = -lam + i * np.log(lam) - log_factorial(i)
        cdf += np.exp(log_p)
    
    return float(pmf), float(cdf)