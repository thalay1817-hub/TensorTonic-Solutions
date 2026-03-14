from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    # PMF
    pmf = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    # CDF
    cdf = 0.0
    for i in range(k + 1):
        cdf += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    return float(pmf), float(cdf)


# Examples
print(binomial_pmf_cdf(5, 0.5, 2))
print(binomial_pmf_cdf(10, 0.3, 0))
print(binomial_pmf_cdf(8, 0.7, 8))