import numpy as np

def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    # Step 1: sort by predicted probabilities
    cal_probs = np.array(cal_probs, float)
    cal_labels = np.array(cal_labels, float)
    order = np.argsort(cal_probs)
    x = cal_probs[order]   # sorted probabilities
    y = cal_labels[order]  # sorted labels

    # Step 2: Pool Adjacent Violators (PAV)
    values = y.copy()
    weights = np.ones(len(y), float)

    i = 0
    while i < len(values) - 1:
        if values[i] > values[i + 1]:   # violation → merge
            # merge backward as needed
            j = i
            while j >= 0 and values[j] > values[j + 1]:
                total_w = weights[j] + weights[j + 1]
                avg = (values[j] * weights[j] + values[j + 1] * weights[j + 1]) / total_w

                values[j] = values[j + 1] = avg
                weights[j] = weights[j + 1] = total_w
                j -= 1

            # restart from previous index
            i = max(j, 0)
        else:
            i += 1

    # Step 3: interpolate calibrated values for new predictions
    calibrated = []
    for q in new_probs:
        q = float(q)

        # clamp
        if q <= x[0]:
            calibrated.append(values[0])
            continue
        if q >= x[-1]:
            calibrated.append(values[-1])
            continue

        # find interval
        idx = np.searchsorted(x, q) - 1
        p_i, p_j = x[idx], x[idx + 1]
        c_i, c_j = values[idx], values[idx + 1]

        # linear interpolation
        t = (q - p_i) / (p_j - p_i)
        calibrated.append(c_i + t * (c_j - c_i))

    # round to 6 decimals like your expected output
    return [round(v, 6) for v in calibrated]