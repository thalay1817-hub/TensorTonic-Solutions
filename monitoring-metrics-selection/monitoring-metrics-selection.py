import math

def compute_monitoring_metrics(system_type, y_true, y_pred):
    n = len(y_true)
    results = []

    if system_type == "classification":
        TP = FP = FN = TN = 0
        
        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                TP += 1
            elif yt == 0 and yp == 1:
                FP += 1
            elif yt == 1 and yp == 0:
                FN += 1
            else:
                TN += 1

        # Metrics with safe division
        accuracy = (TP + TN) / n if n > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        results = [
            ("accuracy", round(accuracy, 4)),
            ("f1", round(f1, 4)),
            ("precision", round(precision, 4)),
            ("recall", round(recall, 4))
        ]

    elif system_type == "regression":
        abs_errors = []
        sq_errors = []
        
        for yt, yp in zip(y_true, y_pred):
            err = yt - yp
            abs_errors.append(abs(err))
            sq_errors.append(err ** 2)

        mae = sum(abs_errors) / n if n > 0 else 0.0
        rmse = math.sqrt(sum(sq_errors) / n) if n > 0 else 0.0

        results = [
            ("mae", round(mae, 4)),
            ("rmse", round(rmse, 4))
        ]

    elif system_type == "ranking":
        # Pair and sort by predicted score descending
        paired = list(zip(y_true, y_pred))
        paired.sort(key=lambda x: x[1], reverse=True)

        top_k = paired[:3]
        relevant_in_top_k = sum(yt for yt, _ in top_k)
        total_relevant = sum(y_true)

        precision_at_3 = relevant_in_top_k / 3 if 3 > 0 else 0.0
        recall_at_3 = (relevant_in_top_k / total_relevant) if total_relevant > 0 else 0.0

        results = [
            ("precision_at_3", round(precision_at_3, 4)),
            ("recall_at_3", round(recall_at_3, 4))
        ]

    # Sort results alphabetically by metric name
    return sorted(results, key=lambda x: x[0])


# Example usage
print(compute_monitoring_metrics(
    "classification",
    [1, 0, 1, 1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 1, 1, 0]
))

print(compute_monitoring_metrics(
    "regression",
    [3.0, 5.0, 2.5, 7.0],
    [2.5, 5.5, 2.0, 8.0]
))