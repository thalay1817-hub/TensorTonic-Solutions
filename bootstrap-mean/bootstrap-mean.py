import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    x = np.array(x, dtype=float)
    N = len(x)
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate bootstrap samples (indices)
    indices = rng.integers(0, N, size=(n_bootstrap, N))
    
    # Compute bootstrap means
    boot_samples = x[indices]
    boot_means = np.mean(boot_samples, axis=1)
    
    # Confidence interval
    alpha = 1 - ci
    lower = np.quantile(boot_means, alpha / 2)
    upper = np.quantile(boot_means, 1 - alpha / 2)
    
    return boot_means, lower, upper