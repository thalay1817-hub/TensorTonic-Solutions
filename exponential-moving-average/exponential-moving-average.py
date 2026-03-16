def exponential_moving_average(values, alpha):
    ema = [float(values[0])]   # EMA[0] = first value

    for i in range(1, len(values)):
        new_ema = alpha * values[i] + (1 - alpha) * ema[-1]
        ema.append(float(new_ema))

    return ema