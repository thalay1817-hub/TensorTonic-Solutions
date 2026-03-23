import math

def evaluate_shadow(production_log, shadow_log, criteria):
    n = len(production_log)
    
    # Accuracy
    prod_correct = sum(1 for p in production_log if p["prediction"] == p["actual"])
    shadow_correct = sum(1 for s in shadow_log if s["prediction"] == s["actual"])
    
    production_accuracy = prod_correct / n
    shadow_accuracy = shadow_correct / n
    accuracy_gain = shadow_accuracy - production_accuracy

    # Shadow latency P95 (nearest-rank method)
    latencies = sorted(s["latency_ms"] for s in shadow_log)
    index = math.ceil(0.95 * n) - 1
    shadow_latency_p95 = latencies[index]

    # Agreement rate
    agreement = sum(
        1 for p, s in zip(production_log, shadow_log)
        if p["prediction"] == s["prediction"]
    )
    agreement_rate = agreement / n

    # Promotion decision
    promote = (
        accuracy_gain >= criteria["min_accuracy_gain"] and
        shadow_latency_p95 <= criteria["max_latency_p95"] and
        agreement_rate >= criteria["min_agreement_rate"]
    )

    return {
        "promote": promote,
        "metrics": {
            "shadow_accuracy": shadow_accuracy,
            "production_accuracy": production_accuracy,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": shadow_latency_p95,
            "agreement_rate": agreement_rate
        }
    }