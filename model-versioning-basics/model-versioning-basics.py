from datetime import datetime

def promote_model(models):
    # Sort models based on:
    # 1. Highest accuracy → descending (-accuracy)
    # 2. Lowest latency → ascending (latency)
    # 3. Latest timestamp → descending (-timestamp)
    
    best_model = sorted(
        models,
        key=lambda m: (
            -m["accuracy"],                      # higher is better
            m["latency"],                        # lower is better
            -datetime.fromisoformat(m["timestamp"]).timestamp()  # newer is better
        )
    )[0]
    
    return best_model["name"]


# Example usage
models1 = [
    {"name": "v1", "accuracy": 0.85, "latency": 120, "timestamp": "2024-01-15"},
    {"name": "v2", "accuracy": 0.91, "latency": 95, "timestamp": "2024-02-20"},
]
print(promote_model(models1))  # v2

models2 = [
    {"name": "v1", "accuracy": 0.90, "latency": 100, "timestamp": "2024-01-10"},
    {"name": "v2", "accuracy": 0.90, "latency": 80, "timestamp": "2024-03-05"},
]
print(promote_model(models2))  # v2