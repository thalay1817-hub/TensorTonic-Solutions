import numpy as np

def streaming_minmax_init(D):
    # Initialize running statistics
    state = {
        "min": np.full(D, np.inf),
        "max": np.full(D, -np.inf)
    }
    return state


def streaming_minmax_update(state, X_batch, eps=1e-8):
    X_batch = np.asarray(X_batch, dtype=float)

    # Update running min and max
    state["min"] = np.minimum(state["min"], np.min(X_batch, axis=0))
    state["max"] = np.maximum(state["max"], np.max(X_batch, axis=0))

    # Normalize batch
    denom = np.maximum(state["max"] - state["min"], eps)
    X_norm = (X_batch - state["min"]) / denom

    return X_norm


# Example 1
state = streaming_minmax_init(D=2)
print(state)

print(streaming_minmax_update(state, [[1,3],[2,1]]))


# Example 2
state = streaming_minmax_init(D=1)
print(state)

print(streaming_minmax_update(state, [[5],[3]]))