def lag_features(series, lags):
    result = []
    max_lag = max(lags)

    # Loop over valid time steps
    for t in range(max_lag, len(series)):
        row = [series[t - lag] for lag in lags]
        result.append(row)

    return result